//! TrieDB transaction and cursor traits for interacting with the trie database.
//!
//! This module provides traits for reading and writing to the trie database,
//! including support for account and storage slot operations.

use alloy_primitives::StorageValue;
use reth_storage_errors::db::DatabaseError;
use triedb::{
    account::Account,
    path::{AddressPath, StoragePath},
};

/// Trait for reading and writing to the trie database.
/// Provides methods for getting account and storage data, and committing changes.
pub trait TrieDbTx: Send + Sync {
    /// Get an account by its address path.
    fn get_account(&self, address_path: AddressPath) -> Result<Option<Account>, DatabaseError>;

    /// Get a storage slot value by its storage path.
    fn get_storage_slot(
        &self,
        storage_path: StoragePath,
    ) -> Result<Option<StorageValue>, DatabaseError>;

    /// Commit any pending changes to the database.
    fn commit(self) -> Result<(), DatabaseError>;
}

/// Trait for read-write operations on the trie database.
/// Extends TrieDbTx with methods for modifying account and storage data.
pub trait TrieDbTxRW: TrieDbTx {
    /// Set an account at the given address path.
    fn set_account(
        &self,
        address_path: AddressPath,
        account: Option<Account>,
    ) -> Result<(), DatabaseError>;
    /// Set a storage slot value at the given storage path.
    fn set_storage_slot(
        &self,
        storage_path: StoragePath,
        value: Option<StorageValue>,
    ) -> Result<(), DatabaseError>;
    /// Apply all pending changes to the database.
    fn apply_changes(&self) -> Result<(), DatabaseError>;
}
