mod root;

pub use root::TrieDbOverlayStateRoot;

use std::{
    collections::BTreeMap,
    fs::create_dir_all,
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
};

use alloy_primitives::{Address, StorageValue, B256, U256};
use alloy_trie::{Nibbles, EMPTY_ROOT_HASH};
use reth_db::triedb::TrieDbTxRW;

use crate::errors::triedb::TrieDBError as RethTrieDBError;
use reth_db_api::triedb::TrieDbTx;
use reth_primitives_traits::Account as RethAccount;
use reth_storage_errors::{
    db::DatabaseError,
    provider::{ProviderError, ProviderResult},
};

use alloy_consensus::constants::KECCAK_EMPTY;
use triedb::{
    account::Account as TrieDBAccount,
    database::{begin_ro, begin_rw},
    overlay::OverlayState,
    path::{AddressPath, StoragePath},
    storage::overlay_root::OverlayedRoot,
    transaction::{Transaction, RO, RW},
    Database,
};

// Account type conversion utilities between Reth and TrieDB
// We can't use From traits due to orphan rules, so we provide conversion functions
pub fn reth_account_to_triedb(reth_account: &RethAccount) -> TrieDBAccount {
    TrieDBAccount {
        nonce: reth_account.nonce,
        balance: reth_account.balance,
        storage_root: EMPTY_ROOT_HASH, // Default empty root for accounts without storage
        code_hash: reth_account.bytecode_hash.unwrap_or(KECCAK_EMPTY),
    }
}

pub fn triedb_account_to_reth(triedb_account: &TrieDBAccount) -> RethAccount {
    RethAccount {
        nonce: triedb_account.nonce,
        balance: triedb_account.balance,
        bytecode_hash: if triedb_account.code_hash == KECCAK_EMPTY {
            None
        } else {
            Some(triedb_account.code_hash)
        },
    }
}

#[derive(Debug, Clone)]
pub struct TrieDbProvider {
    inner: Arc<Database>,
}

impl TrieDbProvider {
    /// Creates a new [`TrieDBProvider`].
    pub fn open(path: impl AsRef<Path>) -> ProviderResult<Self> {
        let database_file_path = path.as_ref().join("triedb.dat");
        let file_exists = database_file_path.exists();
        let db = if file_exists {
            Database::open(database_file_path.to_str().expect("Path must be valid UTF-8"))?
        } else {
            if !path.as_ref().exists() {
                create_dir_all(path).expect("unable to create directory for triedb");
            }
            Database::create_new(database_file_path.to_str().expect("Path must be valid UTF-8"))
                .map_err(|e| {
                    ProviderError::Database(DatabaseError::Other(format!(
                        "TrieDB creation failed: {:?}",
                        e
                    )))
                })?
        };
        Ok(Self { inner: Arc::new(db) })
    }

    /// Returns a read-only TrieDB transaction.
    pub fn tx(&self) -> ProviderResult<TrieDbTransaction> {
        let tx = begin_ro(self.inner.clone())?;
        Ok(TrieDbTransaction { inner: Mutex::new(TrieDbTransactionInner::RO(tx)) })
    }

    /// Returns a read-write TrieDB transaction.
    pub fn tx_mut(&self) -> ProviderResult<TrieDbTransaction> {
        let tx = begin_rw(self.inner.clone())?;
        Ok(TrieDbTransaction { inner: Mutex::new(TrieDbTransactionInner::RW(tx)) })
    }
}

#[derive(Debug)]
enum TrieDbTransactionInner<DB: Deref<Target = Database>> {
    RO(Transaction<DB, RO>),
    RW(Transaction<DB, RW>),
}

// TODO: can be represented as separate read / write traits.
#[derive(Debug)]
pub struct TrieDbTransaction {
    inner: Mutex<TrieDbTransactionInner<Arc<Database>>>,
}

impl TrieDbTransaction {
    pub fn get_account(&self, address_path: AddressPath) -> ProviderResult<Option<TrieDBAccount>> {
        let account = match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RO(tx) => tx.get_account(&address_path)?,
            TrieDbTransactionInner::RW(tx) => tx.get_account(&address_path)?,
        };

        Ok(account)
    }

    pub fn get_storage_slot(
        &self,
        storage_path: StoragePath,
    ) -> ProviderResult<Option<StorageValue>> {
        let storage = match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RO(tx) => tx.get_storage_slot(&storage_path)?,
            TrieDbTransactionInner::RW(tx) => tx.get_storage_slot(&storage_path)?,
        };

        Ok(storage)
    }

    pub fn apply_changes(&self) -> ProviderResult<()> {
        match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RW(_tx) => {
                // TODO: apply changes
                // tx.apply_changes()?;
            }
            TrieDbTransactionInner::RO(_) => {
                // Read-only transactions don't have changes to apply
            }
        }
        Ok(())
    }

    pub fn commit(self) -> ProviderResult<()> {
        match self.inner.into_inner().unwrap() {
            TrieDbTransactionInner::RO(tx) => {
                tx.commit()?;
            }
            TrieDbTransactionInner::RW(tx) => {
                tx.commit()?;
            }
        };

        Ok(())
    }

    pub fn set_account(
        &self,
        hashed_address: B256,
        account: Option<RethAccount>,
    ) -> ProviderResult<(AddressPath, Option<TrieDBAccount>)> {
        // TODO: cleaner way to handle all this? e.g. Supporting `From` for Reth (or Revm) account
        // in TrieDB account.
        let address_path = AddressPath::new(Nibbles::unpack(hashed_address));

        let triedb_account_option: Option<TrieDBAccount> =
            account.as_ref().map(reth_account_to_triedb);
        match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RW(tx) => {
                tx.set_account(address_path.clone(), triedb_account_option.clone())?
            }

            _ => return Err(RethTrieDBError::ReadOnlyWriteError.into()),
        }

        Ok((address_path, triedb_account_option))
    }

    pub fn set_account_address(
        &self,
        address: Address,
        account: Option<RethAccount>,
    ) -> ProviderResult<(AddressPath, Option<TrieDBAccount>)> {
        // TODO: cleaner way to handle all this? e.g. Supporting `From` for Reth (or Revm) account
        // in TrieDB account.
        let address_path = AddressPath::for_address(address);

        let triedb_account_option: Option<TrieDBAccount> =
            account.as_ref().map(reth_account_to_triedb);
        match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RW(tx) => {
                tx.set_account(address_path.clone(), triedb_account_option.clone())?
            }

            _ => return Err(RethTrieDBError::ReadOnlyWriteError.into()),
        }

        Ok((address_path, triedb_account_option))
    }

    pub fn set_storage_slot(
        &self,
        hashed_address: B256,
        key: B256,
        value: Option<U256>,
    ) -> ProviderResult<(StoragePath, Option<StorageValue>)> {
        let address_path = AddressPath::new(Nibbles::unpack(hashed_address));
        let storage_path =
            StoragePath::for_address_path_and_slot_hash(address_path, Nibbles::unpack(key));

        match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RW(tx) => tx.set_storage_slot(storage_path.clone(), value)?,

            _ => return Err(RethTrieDBError::ReadOnlyWriteError.into()),
        }

        Ok((storage_path, value))
    }

    pub fn state_root(&self) -> B256 {
        let state_root = match &*self.inner.lock().unwrap() {
            TrieDbTransactionInner::RO(tx) => tx.state_root(),
            TrieDbTransactionInner::RW(tx) => tx.state_root(),
        };

        state_root
    }

    pub fn compute_root_with_overlay(
        &self,
        overlay_state: OverlayState,
    ) -> ProviderResult<OverlayedRoot> {
        let overlayed_root = match &*self.inner.lock().unwrap() {
            TrieDbTransactionInner::RO(tx) => tx.compute_root_with_overlay(overlay_state)?,
            TrieDbTransactionInner::RW(tx) => tx.compute_root_with_overlay(overlay_state)?,
        };

        Ok(overlayed_root)
    }

    pub fn assert_state_root(&self, expected_state_root: B256) -> ProviderResult<()> {
        let triedb_state_root = self.state_root();
        if triedb_state_root != expected_state_root {
            return Err(RethTrieDBError::AssertionError(format!(
                "state root mismatch: expected: {:?}, got: {:?}",
                expected_state_root, triedb_state_root
            ))
            .into());
        }

        Ok(())
    }

    pub fn assert_db(
        &self,
        address: Address,
        expected_account: TrieDBAccount,
        expected_storage: BTreeMap<U256, U256>,
    ) -> ProviderResult<()> {
        let address_path = AddressPath::for_address(address);
        let db_account = self.get_account(address_path.clone())?;
        if db_account.is_none() {
            return Err(RethTrieDBError::AssertionError(format!(
                "account in database is none. address: {:?}",
                address
            ))
            .into());
        }

        let db_account = db_account.unwrap();

        if db_account.balance != expected_account.balance {
            return Err(RethTrieDBError::AssertionError(format!(
                "account in database has a different balance: expected: {:?}, got: {:?}",
                expected_account.balance, db_account.balance
            ))
            .into());
        }

        if db_account.nonce != expected_account.nonce {
            return Err(RethTrieDBError::AssertionError(format!(
                "account in database has a different nonce: expected: {:?}, got: {:?}",
                expected_account.nonce, db_account.nonce
            ))
            .into());
        }

        if db_account.code_hash != expected_account.code_hash {
            return Err(RethTrieDBError::AssertionError(format!(
                "account in database has a different code hash: expected: {:?}, got: {:?}",
                expected_account.code_hash, db_account.code_hash
            ))
            .into());
        }

        for (slot, value) in &expected_storage {
            let storage_path =
                StoragePath::for_address_path_and_slot(address_path.clone(), (*slot).into());
            let storage_value = self.get_storage_slot(storage_path.clone())?;
            if storage_value.is_none() {
                println!("assert address storage path is {:?}", storage_path.clone());
                return Err(RethTrieDBError::AssertionError(format!(
                    "account storage in database is none. address: {:?}, slot: {:?}",
                    address, slot
                ))
                .into());
            }

            let storage_value = storage_value.unwrap();

            if storage_value != *value {
                return Err(RethTrieDBError::AssertionError(format!(
                    "account storage in database has a different value for slot: {:?}. expected: {:?}, got: {:?}",
                    slot, value, storage_value
                ))
                .into());
            }
        }

        Ok(())
    }

    /*
    TODO: re-enable proofs

    /// Get account with proof using TrieDB's native proof generation
    pub fn get_account_with_proof(
        &self,
        address_path: AddressPath,
    ) -> ProviderResult<Option<(RethAccount, MultiProof)>> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            TrieDbTransactionInner::RO(ref mut tx) => {
                match tx.get_account_with_proof(address_path) {
                    Ok(Some((account, proof))) => {
                        let reth_account = triedb_account_to_reth(account);
                        Ok(Some((reth_account, proof)))
                    }
                    Ok(None) => Ok(None),
                    Err(e) => Err(ProviderError::Database(DatabaseError::Other(e.to_string()))),
                }
            }
            TrieDbTransactionInner::RW(ref mut tx) => {
                match tx.get_account_with_proof(address_path) {
                    Ok(Some((account, proof))) => {
                        let reth_account = triedb_account_to_reth(account);
                        Ok(Some((reth_account, proof)))
                    }
                    Ok(None) => Ok(None),
                    Err(e) => Err(ProviderError::Database(DatabaseError::Other(e.to_string()))),
                }
            }
        }
    }

    /// Get storage with proof using TrieDB's native proof generation
    pub fn get_storage_with_proof(
        &self,
        storage_path: StoragePath,
    ) -> ProviderResult<Option<(StorageValue, MultiProof)>> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            TrieDbTransactionInner::RO(ref mut tx) => {
                match tx.get_storage_with_proof(storage_path) {
                    Ok(result) => Ok(result),
                    Err(e) => Err(ProviderError::Database(DatabaseError::Other(e.to_string()))),
                }
            }
            TrieDbTransactionInner::RW(ref mut tx) => {
                match tx.get_storage_with_proof(storage_path) {
                    Ok(result) => Ok(result),
                    Err(e) => Err(ProviderError::Database(DatabaseError::Other(e.to_string()))),
                }
            }
        }
    }

    */
}

impl TrieDbTx for TrieDbTransaction {
    fn get_account(
        &self,
        address_path: AddressPath,
    ) -> Result<Option<TrieDBAccount>, DatabaseError> {
        self.get_account(address_path).map_err(|e| DatabaseError::Other(e.to_string()))
    }

    fn get_storage_slot(
        &self,
        storage_path: StoragePath,
    ) -> Result<Option<StorageValue>, DatabaseError> {
        self.get_storage_slot(storage_path).map_err(|e| DatabaseError::Other(e.to_string()))
    }

    fn commit(self) -> Result<(), DatabaseError> {
        self.commit().map_err(|e| DatabaseError::Other(e.to_string()))
    }
}

impl TrieDbTxRW for TrieDbTransaction {
    fn set_account(
        &self,
        address_path: AddressPath,
        account: Option<TrieDBAccount>,
    ) -> Result<(), DatabaseError> {
        match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RW(tx) => {
                tx.set_account(address_path, account)
                    .map_err(|e| DatabaseError::Other(e.to_string()))?;
                Ok(())
            }
            _ => {
                Err(DatabaseError::Other("Cannot set account on read-only transaction".to_string()))
            }
        }
    }

    fn set_storage_slot(
        &self,
        storage_path: StoragePath,
        value: Option<StorageValue>,
    ) -> Result<(), DatabaseError> {
        match &mut *self.inner.lock().unwrap() {
            TrieDbTransactionInner::RW(tx) => {
                tx.set_storage_slot(storage_path, value)
                    .map_err(|e| DatabaseError::Other(e.to_string()))?;
                Ok(())
            }
            _ => {
                Err(DatabaseError::Other("Cannot set storage on read-only transaction".to_string()))
            }
        }
    }

    fn apply_changes(&self) -> Result<(), DatabaseError> {
        self.apply_changes().map_err(|e| DatabaseError::Other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Address, B256, U256};
    use alloy_trie::Nibbles;
    use tempfile::TempDir;

    fn create_test_triedb_provider() -> (TrieDbProvider, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let provider =
            TrieDbProvider::open(temp_dir.path()).expect("Failed to create TrieDbProvider");
        (provider, temp_dir)
    }

    #[test]
    fn test_account_type_conversions() {
        // Test Reth -> TrieDB conversion
        let reth_account = RethAccount {
            nonce: 42,
            balance: U256::from(1000000),
            bytecode_hash: Some(B256::from([1u8; 32])),
        };

        let triedb_account = reth_account_to_triedb(&reth_account);
        assert_eq!(triedb_account.nonce, 42);
        assert_eq!(triedb_account.balance, U256::from(1000000));
        assert_eq!(triedb_account.code_hash, B256::from([1u8; 32]));
        assert_eq!(triedb_account.storage_root, EMPTY_ROOT_HASH);

        // Test TrieDB -> Reth conversion
        let converted_back = triedb_account_to_reth(&triedb_account);
        assert_eq!(converted_back.nonce, reth_account.nonce);
        assert_eq!(converted_back.balance, reth_account.balance);
        assert_eq!(converted_back.bytecode_hash, reth_account.bytecode_hash);
    }

    #[test]
    fn test_account_type_conversions_no_code() {
        // Test conversion with no bytecode hash
        let reth_account = RethAccount { nonce: 10, balance: U256::from(500), bytecode_hash: None };

        let triedb_account = reth_account_to_triedb(&reth_account);
        assert_eq!(triedb_account.code_hash, KECCAK_EMPTY);

        let converted_back = triedb_account_to_reth(&triedb_account);
        assert_eq!(converted_back.bytecode_hash, None);
    }

    #[test]
    fn test_triedb_provider_creation() {
        let (_provider, _temp_dir) = create_test_triedb_provider();
        // Provider creation should succeed without panicking
    }

    #[test]
    fn test_triedb_transaction_creation() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Test read-only transaction
        let ro_tx = provider.tx().expect("Failed to create RO transaction");
        assert!(ro_tx.inner.lock().is_ok());

        // Test read-write transaction
        let rw_tx = provider.tx_mut().expect("Failed to create RW transaction");
        assert!(rw_tx.inner.lock().is_ok());
    }

    #[test]
    fn test_triedb_account_operations() {
        let (provider, _temp_dir) = create_test_triedb_provider();
        let tx = provider.tx_mut().expect("Failed to create RW transaction");

        // Test account operations
        let address = Address::from([0x42; 20]);
        let account = RethAccount { nonce: 1, balance: U256::from(1000), bytecode_hash: None };

        // Set account
        let result = tx.set_account_address(address, Some(account));
        assert!(result.is_ok());

        // Apply changes to make them visible in the trie
        tx.apply_changes().expect("Failed to apply changes");

        // Get account - should be visible after apply_changes
        let address_path = AddressPath::for_address(address);
        let retrieved = tx.get_account(address_path).expect("Failed to get account");
        assert!(retrieved.is_some(), "Account should be visible after applying changes");
        let retrieved_account = retrieved.unwrap();
        assert_eq!(retrieved_account.nonce, account.nonce);
        assert_eq!(retrieved_account.balance, account.balance);
    }

    #[test]
    fn test_triedb_storage_operations() {
        let (provider, _temp_dir) = create_test_triedb_provider();
        let tx = provider.tx_mut().expect("Failed to create RW transaction");

        // Set up account first
        let address = Address::from([0x42; 20]);
        let account = RethAccount { nonce: 1, balance: U256::from(1000), bytecode_hash: None };
        tx.set_account_address(address, Some(account)).expect("Failed to set account");

        // Test storage operations
        let storage_key = B256::from([0x01; 32]);
        let storage_value = U256::from(0x1234);

        // Apply changes for account to be visible
        tx.apply_changes().expect("Failed to apply account changes");

        // Set storage - use keccak hash of address
        let hashed_address = alloy_primitives::keccak256(address.0);
        tx.set_storage_slot(hashed_address, storage_key, Some(storage_value))
            .expect("Failed to set storage slot");

        // Apply changes to make storage visible in the trie
        tx.apply_changes().expect("Failed to apply storage changes");

        // Get storage - should be visible after apply_changes
        // Use the same path construction as in set_storage_slot
        let address_path = AddressPath::new(Nibbles::unpack(hashed_address));
        let storage_path =
            StoragePath::for_address_path_and_slot_hash(address_path, Nibbles::unpack(storage_key));
        let retrieved = tx.get_storage_slot(storage_path).expect("Failed to get storage");
        assert!(retrieved.is_some(), "Storage should be visible after applying changes");
        assert_eq!(retrieved.unwrap(), storage_value);
    }
}
