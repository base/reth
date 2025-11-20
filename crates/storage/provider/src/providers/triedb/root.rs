use std::time::Instant;

use alloy_primitives::{
    map::{B256Map, HashSet},
    B256,
};
use alloy_trie::Nibbles;
use reth_trie::{
    updates::{StorageTrieUpdates, TrieUpdates},
    HashedPostState, TrieInput,
};
use reth_trie_common::{
    prefix_set::TriePrefixSets,
};
use triedb::overlay::{OverlayState, OverlayStateMut, OverlayValue};

use crate::providers::triedb::{reth_account_to_triedb, TrieDbTransaction};
use reth_storage_errors::provider::ProviderError;

pub struct TrieDbOverlayStateRoot {
    tx: TrieDbTransaction,
    input_nodes: TrieUpdates,
    input_state: HashedPostState,
    input_prefix_sets: TriePrefixSets,
}

impl TrieDbOverlayStateRoot {
    pub fn new(tx: TrieDbTransaction, input: TrieInput) -> Self {
        Self {
            tx,
            input_nodes: input.nodes,
            input_state: input.state,
            input_prefix_sets: input.prefix_sets.freeze(),
        }
    }

    pub fn incremental_root(self) -> Result<B256, ProviderError> {
        let (root, _) = self.calculate(false)?;
        Ok(root)
    }

    pub fn incremental_root_with_updates(self) -> Result<(B256, TrieUpdates), ProviderError> {
        self.calculate(true)
    }

    fn calculate(self, retain_updates: bool) -> Result<(B256, TrieUpdates), ProviderError> {
        // println!("TrieDBOverlayStateRoot::calculate: prefix_sets: {:?}", self.input_prefix_sets);
        let start_time = Instant::now();
        let (overlay_state, removed_keys, mut removed_storage_keys) = build_overlay_state(
            self.input_nodes,
            self.input_state,
            self.input_prefix_sets,
        );
        let overlay_time = start_time.elapsed();
        // println!("TrieDBOverlayStateRoot::calculate: overlay_state: {:?}", overlay_state);
        // println!("TrieDBOverlayStateRoot::calculate: removed_keys: {:?}", removed_keys);
        // println!("TrieDBOverlayStateRoot::calculate: removed_storage_keys: {:?}",
        // removed_storage_keys);
        let overlayed_root = self.tx.compute_root_with_overlay(overlay_state)?;
        let overlayed_root_time = start_time.elapsed();
        let (root, branch_updates, storage_updates) = (
            overlayed_root.root,
            overlayed_root.updated_branch_nodes,
            overlayed_root.storage_branch_updates,
        );
        // println!("TrieDBOverlayStateRoot::calculate: root: {:?}", root);
        // println!("TrieDBOverlayStateRoot::calculate: branch_updates: {:?}", branch_updates);
        // println!("TrieDBOverlayStateRoot::calculate: storage_updates: {:?}", storage_updates);
        let mut trie_updates = TrieUpdates::default();
        trie_updates.account_nodes.extend(branch_updates);
        for (hashed_address, storage_updates) in storage_updates {
            trie_updates.insert_storage_updates(
                hashed_address,
                StorageTrieUpdates {
                    is_deleted: false,
                    storage_nodes: storage_updates,
                    removed_nodes: removed_storage_keys.remove(&hashed_address).unwrap_or_default(),
                },
            );
        }
        trie_updates.removed_nodes.extend(removed_keys);
        let trie_updates_time = start_time.elapsed();
        tracing::debug!(
            target: "TrieDBOverlayStateRoot::calculate",
            overlay_time = ?overlay_time,
            "Computed TrieDB overlay"
        );
        tracing::debug!(
            target: "TrieDBOverlayStateRoot::calculate",
            overlayed_root_time = ?overlayed_root_time,
            "Computed TrieDB overlayed root"
        );
        tracing::debug!(
            target: "TrieDBOverlayStateRoot::calculate",
            trie_updates_time = ?trie_updates_time,
            "Computed TrieDB trie updates"
        );
        Ok((root, trie_updates))
    }
}

fn build_overlay_state(
    input_nodes: TrieUpdates,
    input_state: HashedPostState,
    mut input_prefix_sets: TriePrefixSets,
) -> (OverlayState, HashSet<Nibbles>, B256Map<HashSet<Nibbles>>) {
    let mut removed_keys = HashSet::default();
    let mut removed_storage_keys: B256Map<HashSet<Nibbles>> = B256Map::default();
    let input_nodes = input_nodes.into_sorted();
    let mut overlay_mut = OverlayStateMut::with_capacity(input_nodes.account_nodes.len() * 16 + input_nodes.storage_tries.len() * 16);
    for (key, branch) in input_nodes.account_nodes {
        if input_prefix_sets.account_prefix_set.contains(&key) {
            removed_keys.insert(key);
            continue;
        }
        let mut hash_idx = 0;
        let mut path = key;
        for i in 0..16 {
            if branch.hash_mask.is_bit_set(i) {
                path.push(i);
                overlay_mut
                    .insert(path.clone().into(), Some(OverlayValue::Hash(branch.hashes[hash_idx])));
                hash_idx += 1;
                path.pop();
            }
        }
    }
    for (account, storage_updates) in input_nodes.storage_tries {
        let mut storage_prefix_set =
            input_prefix_sets.storage_prefix_sets.get_mut(&account);
        for (key, branch) in storage_updates.storage_nodes {
            if let Some(ref mut prefix_set) = storage_prefix_set {
                if prefix_set.contains(&key) {
                    removed_storage_keys.entry(account).or_default().insert(key);
                    continue;
                }
            }
            let mut hash_idx = 0;
            let mut path = Nibbles::unpack(account).join(&key);
            for i in 0..16 {
                if branch.hash_mask.is_bit_set(i) {
                    path.push(i);
                    overlay_mut.insert(
                        path.clone().into(),
                        Some(OverlayValue::Hash(branch.hashes[hash_idx])),
                    );
                    hash_idx += 1;
                    path.pop();
                }
            }
        }
    }
    for (key, node) in input_state.accounts {
        if let Some(account) = node {
            overlay_mut.insert(
                Nibbles::unpack(key).into(),
                Some(OverlayValue::Account(reth_account_to_triedb(&account))),
            );
        } else {
            overlay_mut.insert(Nibbles::unpack(key).into(), None);
        }
    }
    for (account, storage_updates) in input_state.storages {
        let account_path = Nibbles::unpack(account);
        for (key, value) in storage_updates.storage.into_iter() {
            overlay_mut.insert(
                account_path.join(&Nibbles::unpack(key)).into(),
                Some(OverlayValue::Storage(value)),
            );
        }
    }
    (overlay_mut.freeze(), removed_keys, removed_storage_keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::triedb::TrieDbProvider;
    use alloy_primitives::{Address, B256, U256};
    use reth_primitives_traits::Account as RethAccount;
    use reth_trie::{HashedPostState, HashedStorage, EMPTY_ROOT_HASH};
    use tempfile::TempDir;
    use triedb::path::AddressPath;

    fn create_test_triedb_provider() -> (TrieDbProvider, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let provider =
            TrieDbProvider::open(temp_dir.path()).expect("Failed to create TrieDbProvider");
        (provider, temp_dir)
    }

    #[test]
    fn test_state_root_operations() {
        let (provider, _temp_dir) = create_test_triedb_provider();
        let tx = provider.tx().expect("Failed to create transaction");

        // Test state root retrieval
        let state_root = tx.state_root();
        assert_eq!(state_root.len(), 32); // Should be a valid B256

        // Test state root assertion (should pass for empty trie)
        let assert_result = tx.assert_state_root(state_root);
        assert!(assert_result.is_ok());

        // Test state root assertion with wrong root (should fail)
        let wrong_root = B256::from([0xFF; 32]);
        let assert_result = tx.assert_state_root(wrong_root);
        assert!(assert_result.is_err());
    }

    #[test]
    fn test_triedb_state_root_with_overlays() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed a small amount of initial data into TrieDB
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        // Use known hashed addresses for predictable trie structure
        let hashed_addr1 = B256::from([0x01; 32]); // Simple known hash 1
        let hashed_addr2 = B256::from([0x02; 32]); // Simple known hash 2

        let account1 = RethAccount {
            nonce: 1,
            balance: U256::from(1000),
            bytecode_hash: Some(B256::from([0xAA; 32])),
        };

        let account2 = RethAccount { nonce: 2, balance: U256::from(2000), bytecode_hash: None };

        // Set accounts using the hashed addresses directly
        initial_tx
            .set_account(hashed_addr1, Some(account1.clone()))
            .expect("Failed to set initial account 1");
        initial_tx
            .set_account(hashed_addr2, Some(account2.clone()))
            .expect("Failed to set initial account 2");
        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root with seeded data
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();
        // Phase 2: Create simple overlay state
        let mut hashed_state = HashedPostState::default();

        // Modify account1 - change nonce and balance (using the same known hashed address)
        let modified_account1 = RethAccount {
            nonce: 10,                                   // Changed from 1 to 10
            balance: U256::from(5000),                   // Changed from 1000 to 5000
            bytecode_hash: Some(B256::from([0xBB; 32])), // Changed bytecode
        };
        hashed_state.accounts.insert(hashed_addr1, Some(modified_account1.clone()));

        // Phase 3: Calculate state root WITH overlay
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");

        // Verify overlay state root is different from baseline
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "Overlay state root should differ from baseline"
        );

        // Phase 4: Commit the overlay changes to TrieDB (making it authoritative)
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        // Apply the account modification using the hashed address
        commit_tx
            .set_account(hashed_addr1, Some(modified_account1))
            .expect("Failed to apply account change");
        commit_tx.commit().expect("Failed to commit overlay changes");

        // Phase 5: Get state root from TrieDB after committing overlay (authoritative)
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();

        // Phase 6: Calculate state root again with empty overlay (should match committed)
        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate state root with empty overlay");

        // Verify the state roots match
        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with overlay should match TrieDB state root after committing overlay"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match calculation with empty overlay"
        );

        // Verify final state root is different from baseline
        assert_ne!(
            committed_state_root, baseline_state_root,
            "Final state root should be different from baseline after changes"
        );
    }

    #[test]
    fn test_state_root_with_known_values() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Create a simple known state
        let mut hashed_state = HashedPostState::default();

        // Use a simple address that will hash to a known value
        let address = Address::ZERO;
        let hashed_address = alloy_primitives::keccak256(address.0);

        let account = RethAccount { nonce: 0, balance: U256::ZERO, bytecode_hash: None };
        hashed_state.accounts.insert(hashed_address, Some(account));

        // Calculate state root
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state),
        );
        let state_root_result = state_root_calc.incremental_root();

        assert!(state_root_result.is_ok());
        let state_root = state_root_result.unwrap();

        // For a single empty account, we should get a predictable root
        assert_eq!(state_root.len(), 32); // Valid B256
        assert_ne!(state_root, EMPTY_ROOT_HASH); // Should not be empty with an account

        // The exact root value depends on the trie implementation, but should be consistent
        println!("State root for empty account: {:?}", state_root);
    }

    #[test]
    fn test_state_root_simple_debug() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed just 3 accounts into TrieDB
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let accounts_data = vec![
            (
                B256::from([0x01; 32]),
                RethAccount {
                    nonce: 1,
                    balance: U256::from(1000),
                    bytecode_hash: Some(B256::from([0xA1; 32])),
                },
            ),
            (
                B256::from([0x02; 32]),
                RethAccount {
                    nonce: 2,
                    balance: U256::from(2000),
                    bytecode_hash: Some(B256::from([0xA2; 32])),
                },
            ),
            (
                B256::from([0x03; 32]),
                RethAccount { nonce: 3, balance: U256::from(3000), bytecode_hash: None },
            ),
        ];

        for (hashed_addr, account) in &accounts_data {
            initial_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to set initial account");
        }
        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();
        println!("Baseline state root: {:?}", baseline_state_root);

        // Phase 2: Create overlay modifying just ONE account
        let mut hashed_state = HashedPostState::default();

        // Modify only account 1
        hashed_state.accounts.insert(
            B256::from([0x01; 32]),
            Some(RethAccount {
                nonce: 10,                                   // Changed from 1
                balance: U256::from(10000),                  // Changed from 1000
                bytecode_hash: Some(B256::from([0xB1; 32])), // Changed bytecode
            }),
        );

        let trie_input = TrieInput::from_state(hashed_state);

        // Phase 3: Calculate state root with overlay
        println!(
            "DEBUG: Starting overlay state root calculation with {} accounts in overlay",
            trie_input.state.accounts.len()
        );
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            trie_input,
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");
        println!("Overlay state root: {:?}", overlay_state_root);

        // Verify overlay state root differs from baseline
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "Overlay state root should differ from baseline"
        );

        // Phase 4: Commit overlay to TrieDB (step by step)
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        // Apply just the single modification
        commit_tx
            .set_account(
                B256::from([0x01; 32]),
                Some(RethAccount {
                    nonce: 10,
                    balance: U256::from(10000),
                    bytecode_hash: Some(B256::from([0xB1; 32])),
                }),
            )
            .expect("Failed to apply account change");
        commit_tx.commit().expect("Failed to commit overlay changes");

        // Phase 5: Verify consistency with detailed debugging
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();
        println!("Committed state root: {:?}", committed_state_root);

        // Debug what's actually in TrieDB after commit
        // let debug_state_root = debug_triedb_contents(&provider, "TrieDB contents after commit");

        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate empty overlay state root");
        println!("Empty overlay state root: {:?}", empty_overlay_state_root);

        // All three calculations should match
        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with overlay should match TrieDB after commit"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match empty overlay calculation"
        );
    }

    #[test]
    fn test_debug_missing_account_2() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed 3 accounts into TrieDB
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let accounts_data = vec![
            (
                B256::from([0x01; 32]),
                RethAccount {
                    nonce: 1,
                    balance: U256::from(1000),
                    bytecode_hash: Some(B256::from([0xA1; 32])),
                },
            ),
            (
                B256::from([0x02; 32]),
                RethAccount {
                    nonce: 2,
                    balance: U256::from(2000),
                    bytecode_hash: Some(B256::from([0xA2; 32])),
                },
            ),
            (
                B256::from([0x03; 32]),
                RethAccount { nonce: 3, balance: U256::from(3000), bytecode_hash: None },
            ),
        ];

        for (hashed_addr, account) in &accounts_data {
            initial_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to set initial account");
        }
        initial_tx.commit().expect("Failed to commit initial data");

        // Phase 2: Create overlay modifying only account 3 (not 1 or 2)
        let mut hashed_state = HashedPostState::default();

        // Modify only account 3
        hashed_state.accounts.insert(
            B256::from([0x03; 32]),
            Some(RethAccount {
                nonce: 30,                                   // Changed from 3
                balance: U256::from(30000),                  // Changed from 3000
                bytecode_hash: Some(B256::from([0xB3; 32])), // Added bytecode
            }),
        );

        // Phase 3: Calculate state root with overlay
        println!("=== Testing overlay with only account 3 modified ===");
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");
        println!("Overlay state root: {:?}", overlay_state_root);

        // This should include accounts 1, 2 (from TrieDB) and 3 (from overlay)
        // If account 2 is missing, we'll see only 2 accounts instead of 3
    }

    #[test]
    fn test_debug_exact_failing_scenario() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed exactly 5 accounts like in the failing test
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let accounts_data = vec![
            (
                B256::from([0x01; 32]),
                RethAccount {
                    nonce: 1,
                    balance: U256::from(1000),
                    bytecode_hash: Some(B256::from([0xA1; 32])),
                },
            ),
            (
                B256::from([0x02; 32]),
                RethAccount {
                    nonce: 2,
                    balance: U256::from(2000),
                    bytecode_hash: Some(B256::from([0xA2; 32])),
                },
            ),
            (
                B256::from([0x03; 32]),
                RethAccount { nonce: 3, balance: U256::from(3000), bytecode_hash: None },
            ),
            (
                B256::from([0x04; 32]),
                RethAccount {
                    nonce: 4,
                    balance: U256::from(4000),
                    bytecode_hash: Some(B256::from([0xA4; 32])),
                },
            ),
            (
                B256::from([0x05; 32]),
                RethAccount { nonce: 5, balance: U256::from(5000), bytecode_hash: None },
            ),
        ];

        for (hashed_addr, account) in &accounts_data {
            initial_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to set initial account");
        }
        initial_tx.commit().expect("Failed to commit initial data");

        // Phase 2: Create overlay modifying accounts 1, 3, and 5 (exactly like the failing test)
        let mut hashed_state = HashedPostState::default();

        hashed_state.accounts.insert(
            B256::from([0x01; 32]),
            Some(RethAccount {
                nonce: 10,                                   // Changed from 1
                balance: U256::from(10000),                  // Changed from 1000
                bytecode_hash: Some(B256::from([0xB1; 32])), // Changed bytecode
            }),
        );

        hashed_state.accounts.insert(
            B256::from([0x03; 32]),
            Some(RethAccount {
                nonce: 30,                                   // Changed from 3
                balance: U256::from(30000),                  // Changed from 3000
                bytecode_hash: Some(B256::from([0xB3; 32])), // Added bytecode
            }),
        );

        hashed_state.accounts.insert(
            B256::from([0x05; 32]),
            Some(RethAccount {
                nonce: 50,                  // Changed from 5
                balance: U256::from(50000), // Changed from 5000
                bytecode_hash: None,        // Unchanged
            }),
        );

        // Phase 3: Calculate state root with overlay
        println!("=== Testing exact failing scenario: 5 accounts, modify 1,3,5 ===");
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");
        println!("Overlay state root: {:?}", overlay_state_root);

        // This should include ALL 5 accounts:
        // 1 (from overlay), 2 (from TrieDB), 3 (from overlay), 4 (from TrieDB), 5 (from overlay)
        // If account 2 is missing like in the failure, we'll see only 4 accounts
    }

    #[test]
    fn test_state_root_multiple_account_modifications() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed multiple accounts into TrieDB
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        // Create 5 initial accounts with known hashed addresses
        let accounts_data = vec![
            (
                B256::from([0x01; 32]),
                RethAccount {
                    nonce: 1,
                    balance: U256::from(1000),
                    bytecode_hash: Some(B256::from([0xA1; 32])),
                },
            ),
            (
                B256::from([0x02; 32]),
                RethAccount {
                    nonce: 2,
                    balance: U256::from(2000),
                    bytecode_hash: Some(B256::from([0xA2; 32])),
                },
            ),
            (
                B256::from([0x03; 32]),
                RethAccount { nonce: 3, balance: U256::from(3000), bytecode_hash: None },
            ),
            (
                B256::from([0x04; 32]),
                RethAccount {
                    nonce: 4,
                    balance: U256::from(4000),
                    bytecode_hash: Some(B256::from([0xA4; 32])),
                },
            ),
            (
                B256::from([0x05; 32]),
                RethAccount { nonce: 5, balance: U256::from(5000), bytecode_hash: None },
            ),
        ];

        for (hashed_addr, account) in &accounts_data {
            initial_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to set initial account");
        }
        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();

        // Phase 2: Create overlay modifying multiple accounts
        let mut hashed_state = HashedPostState::default();

        // Modify accounts 1, 3, and 5
        hashed_state.accounts.insert(
            B256::from([0x01; 32]),
            Some(RethAccount {
                nonce: 10,                                   // Changed from 1
                balance: U256::from(10000),                  // Changed from 1000
                bytecode_hash: Some(B256::from([0xB1; 32])), // Changed bytecode
            }),
        );

        hashed_state.accounts.insert(
            B256::from([0x03; 32]),
            Some(RethAccount {
                nonce: 30,                                   // Changed from 3
                balance: U256::from(30000),                  // Changed from 3000
                bytecode_hash: Some(B256::from([0xB3; 32])), // Added bytecode
            }),
        );

        hashed_state.accounts.insert(
            B256::from([0x05; 32]),
            Some(RethAccount {
                nonce: 50,                  // Changed from 5
                balance: U256::from(50000), // Changed from 5000
                bytecode_hash: None,        // Unchanged
            }),
        );

        // Phase 3: Calculate state root with overlays
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");

        // Verify overlay state root differs from baseline
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "Overlay state root should differ from baseline"
        );

        // Phase 4: Commit overlays to TrieDB (apply changes individually)
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        commit_tx
            .set_account(
                B256::from([0x01; 32]),
                Some(RethAccount {
                    nonce: 10,
                    balance: U256::from(10000),
                    bytecode_hash: Some(B256::from([0xB1; 32])),
                }),
            )
            .expect("Failed to apply account 1 change");

        commit_tx
            .set_account(
                B256::from([0x03; 32]),
                Some(RethAccount {
                    nonce: 30,
                    balance: U256::from(30000),
                    bytecode_hash: Some(B256::from([0xB3; 32])),
                }),
            )
            .expect("Failed to apply account 3 change");

        commit_tx
            .set_account(
                B256::from([0x05; 32]),
                Some(RethAccount { nonce: 50, balance: U256::from(50000), bytecode_hash: None }),
            )
            .expect("Failed to apply account 5 change");

        commit_tx.commit().expect("Failed to commit overlay changes");

        // Phase 5: Verify consistency
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();

        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate empty overlay state root");

        // All three calculations should match
        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with overlay should match TrieDB after commit"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match empty overlay calculation"
        );
    }

    #[test]
    fn test_state_root_account_deletions() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed accounts into TrieDB
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let accounts_to_delete = vec![
            (
                B256::from([0x10; 32]),
                RethAccount {
                    nonce: 1,
                    balance: U256::from(1000),
                    bytecode_hash: Some(B256::from([0xD1; 32])),
                },
            ),
            (
                B256::from([0x20; 32]),
                RethAccount { nonce: 2, balance: U256::from(2000), bytecode_hash: None },
            ),
        ];

        let accounts_to_keep = vec![
            (
                B256::from([0x30; 32]),
                RethAccount {
                    nonce: 3,
                    balance: U256::from(3000),
                    bytecode_hash: Some(B256::from([0xC1; 32])),
                },
            ),
            (
                B256::from([0x40; 32]),
                RethAccount { nonce: 4, balance: U256::from(4000), bytecode_hash: None },
            ),
        ];

        for (hashed_addr, account) in accounts_to_delete.iter().chain(accounts_to_keep.iter()) {
            initial_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to set initial account");
        }

        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();

        // Phase 2: Create overlay deleting some accounts
        let mut hashed_state = HashedPostState::default();

        // Delete the first two accounts (None = deletion)
        for (hashed_addr, _) in &accounts_to_delete {
            hashed_state.accounts.insert(*hashed_addr, None);
        }

        // Phase 3: Calculate state root with deletions
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");

        // Verify state root changed
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "State root should change after deletions"
        );

        // Phase 4: Commit deletions to TrieDB
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        for (hashed_addr, _) in &accounts_to_delete {
            commit_tx.set_account(*hashed_addr, None).expect("Failed to delete account");
        }
        commit_tx.commit().expect("Failed to commit deletion changes");

        // Phase 5: Verify consistency
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();

        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate empty overlay state root");

        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with deletions should match TrieDB after commit"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match empty overlay calculation"
        );
    }

    #[test]
    fn test_state_root_new_account_additions() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed some initial accounts
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let existing_accounts = vec![
            (
                B256::from([0x11; 32]),
                RethAccount {
                    nonce: 1,
                    balance: U256::from(1000),
                    bytecode_hash: Some(B256::from([0xE1; 32])),
                },
            ),
            (
                B256::from([0x22; 32]),
                RethAccount { nonce: 2, balance: U256::from(2000), bytecode_hash: None },
            ),
        ];

        for (hashed_addr, account) in &existing_accounts {
            initial_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to set initial account");
        }

        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();

        // Phase 2: Create overlay adding new accounts
        let mut hashed_state = HashedPostState::default();

        // Add new accounts that don't exist in TrieDB
        let new_accounts = vec![
            (
                B256::from([0x33; 32]),
                RethAccount {
                    nonce: 10,
                    balance: U256::from(10000),
                    bytecode_hash: Some(B256::from([0xC1; 32])),
                },
            ),
            (
                B256::from([0x44; 32]),
                RethAccount { nonce: 20, balance: U256::from(20000), bytecode_hash: None },
            ),
            (
                B256::from([0x55; 32]),
                RethAccount {
                    nonce: 30,
                    balance: U256::from(30000),
                    bytecode_hash: Some(B256::from([0xC3; 32])),
                },
            ),
        ];

        for (hashed_addr, account) in &new_accounts {
            hashed_state.accounts.insert(*hashed_addr, Some(account.clone()));
        }

        // Phase 3: Calculate state root with additions
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");

        // Verify state root changed
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "State root should change after additions"
        );

        // Phase 4: Commit additions to TrieDB
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        for (hashed_addr, account) in &new_accounts {
            commit_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to add new account");
        }
        commit_tx.commit().expect("Failed to commit addition changes");

        // Phase 5: Verify consistency
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();

        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate empty overlay state root");

        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with additions should match TrieDB after commit"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match empty overlay calculation"
        );
    }

    #[test]
    fn test_state_root_mixed_operations() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed baseline accounts into TrieDB
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let baseline_accounts = vec![
            (
                B256::from([0x01; 32]),
                RethAccount {
                    nonce: 1,
                    balance: U256::from(1000),
                    bytecode_hash: Some(B256::from([0xC1; 32])),
                },
            ),
            (
                B256::from([0x02; 32]),
                RethAccount { nonce: 2, balance: U256::from(2000), bytecode_hash: None },
            ),
            (
                B256::from([0x03; 32]),
                RethAccount {
                    nonce: 3,
                    balance: U256::from(3000),
                    bytecode_hash: Some(B256::from([0xC3; 32])),
                },
            ),
        ];

        for (hashed_addr, account) in &baseline_accounts {
            initial_tx
                .set_account(*hashed_addr, Some(account.clone()))
                .expect("Failed to set baseline account");
        }
        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();

        // Phase 2: Create overlay with mixed operations
        let mut hashed_state = HashedPostState::default();

        // UPDATE: Modify existing account 0x01
        hashed_state.accounts.insert(
            B256::from([0x01; 32]),
            Some(RethAccount {
                nonce: 100,                                  // Changed from 1
                balance: U256::from(100000),                 // Changed from 1000
                bytecode_hash: Some(B256::from([0xCF; 32])), // Changed bytecode
            }),
        );

        // DELETE: Remove existing account 0x02
        hashed_state.accounts.insert(B256::from([0x02; 32]), None);

        // KEEP UNCHANGED: Account 0x03 not in overlay (should remain unchanged)

        // ADD: Insert new accounts 0x04 and 0x05
        hashed_state.accounts.insert(
            B256::from([0x04; 32]),
            Some(RethAccount {
                nonce: 40,
                balance: U256::from(40000),
                bytecode_hash: Some(B256::from([0xC4; 32])),
            }),
        );

        hashed_state.accounts.insert(
            B256::from([0x05; 32]),
            Some(RethAccount { nonce: 50, balance: U256::from(50000), bytecode_hash: None }),
        );

        // Phase 3: Calculate state root with mixed operations
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state.clone()),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");

        // Verify state root changed
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "State root should change after mixed operations"
        );

        // Phase 4: Commit mixed operations to TrieDB
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        // Apply all overlay changes
        for (hashed_addr, account_opt) in hashed_state.accounts.iter() {
            commit_tx
                .set_account(*hashed_addr, account_opt.clone())
                .expect("Failed to apply mixed operation");
        }
        commit_tx.commit().expect("Failed to commit mixed operation changes");

        // Phase 5: Verify consistency
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();

        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate empty overlay state root");

        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with mixed operations should match TrieDB after commit"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match empty overlay calculation"
        );
    }

    #[test]
    fn test_state_root_large_dataset() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Seed large number of accounts (100 accounts)
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let num_baseline_accounts = 50;
        for i in 0..num_baseline_accounts {
            let mut addr_bytes = [0u8; 32];
            addr_bytes[31] = i as u8; // Use last byte for differentiation
            let hashed_addr = B256::from(addr_bytes);

            let account = RethAccount {
                nonce: i as u64,
                balance: U256::from(i * 1000),
                bytecode_hash: if i % 3 == 0 { Some(B256::from([i as u8; 32])) } else { None },
            };

            initial_tx
                .set_account(hashed_addr, Some(account))
                .expect("Failed to set baseline account");
        }
        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();

        // Phase 2: Create overlay modifying subset of accounts
        let mut hashed_state = HashedPostState::default();

        // Modify every 5th account (10 total modifications)
        for i in (0..num_baseline_accounts).step_by(5) {
            let mut addr_bytes = [0u8; 32];
            addr_bytes[31] = i as u8;
            let hashed_addr = B256::from(addr_bytes);

            hashed_state.accounts.insert(
                hashed_addr,
                Some(RethAccount {
                    nonce: (i + 100) as u64,                                // Changed
                    balance: U256::from((i + 100) * 1000),                  // Changed
                    bytecode_hash: Some(B256::from([(i + 200) as u8; 32])), // Changed
                }),
            );
        }

        // Add some new accounts
        for i in num_baseline_accounts..(num_baseline_accounts + 20) {
            let mut addr_bytes = [0u8; 32];
            addr_bytes[31] = i as u8;
            let hashed_addr = B256::from(addr_bytes);

            hashed_state.accounts.insert(
                hashed_addr,
                Some(RethAccount {
                    nonce: i as u64,
                    balance: U256::from(i * 2000),
                    bytecode_hash: if i % 2 == 0 { Some(B256::from([i as u8; 32])) } else { None },
                }),
            );
        }

        // Phase 3: Calculate state root with large dataset
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state.clone()),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");

        // Verify state root changed
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "State root should change after large dataset operations"
        );

        // Phase 4: Commit changes to TrieDB
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        for (hashed_addr, account_opt) in hashed_state.accounts.iter() {
            commit_tx
                .set_account(*hashed_addr, account_opt.clone())
                .expect("Failed to apply large dataset change");
        }
        commit_tx.commit().expect("Failed to commit large dataset changes");

        // Phase 5: Verify consistency
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();

        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate empty overlay state root");

        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with large dataset should match TrieDB after commit"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match empty overlay calculation"
        );
    }

    #[test]
    fn test_state_root_with_storage_overlay() {
        let (provider, _temp_dir) = create_test_triedb_provider();

        // Phase 1: Set up account with storage in TrieDB
        let initial_tx = provider.tx_mut().expect("Failed to create initial RW transaction");

        let hashed_address = B256::from([0x01; 32]);
        let account = RethAccount {
            nonce: 1,
            balance: U256::from(1000),
            bytecode_hash: Some(B256::from([0xAB; 32])),
        };

        initial_tx
            .set_account(hashed_address, Some(account.clone()))
            .expect("Failed to set account");

        // Add some initial storage
        let storage_key1 = B256::from([0x01; 32]);
        let storage_value1 = U256::from(100);
        let storage_key2 = B256::from([0x02; 32]);
        let storage_value2 = U256::from(200);

        initial_tx
            .set_storage_slot(hashed_address, storage_key1, Some(storage_value1))
            .expect("Failed to set storage 1");
        initial_tx
            .set_storage_slot(hashed_address, storage_key2, Some(storage_value2))
            .expect("Failed to set storage 2");
        initial_tx.commit().expect("Failed to commit initial data");

        // Get baseline state root
        let baseline_tx = provider.tx().expect("Failed to create baseline RO transaction");
        let baseline_state_root = baseline_tx.state_root();

        // Phase 2: Create overlay with storage modifications
        let mut hashed_state = HashedPostState::default();

        // Keep the account the same, but modify storage
        hashed_state.accounts.insert(hashed_address, Some(account.clone()));

        // Create storage overlay
        let mut storage = HashedStorage::default();
        // Modify existing storage value
        storage.storage.insert(storage_key1, U256::from(150)); // Changed from 100 to 150
                                                               // Add new storage value
        let storage_key3 = B256::from([0x03; 32]);
        storage.storage.insert(storage_key3, U256::from(300));

        hashed_state.storages.insert(hashed_address, storage);

        // Phase 3: Calculate state root with storage overlay
        let state_root_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::from_state(hashed_state.clone()),
        );
        let overlay_state_root =
            state_root_calc.incremental_root().expect("Failed to calculate overlay state root");

        // Verify state root changed due to storage modifications
        assert_ne!(
            overlay_state_root, baseline_state_root,
            "State root should change after storage modifications"
        );

        // Phase 4: Commit storage changes to TrieDB
        let commit_tx = provider.tx_mut().expect("Failed to create RW transaction for commit");

        // Apply storage modifications
        commit_tx
            .set_storage_slot(hashed_address, storage_key1, Some(U256::from(150)))
            .expect("Failed to modify storage 1");
        commit_tx
            .set_storage_slot(hashed_address, storage_key3, Some(U256::from(300)))
            .expect("Failed to add storage 3");

        commit_tx.commit().expect("Failed to commit storage changes");

        // Phase 5: Verify consistency
        let final_tx = provider.tx().expect("Failed to create final RO transaction");
        let committed_state_root = final_tx.state_root();
        let account = final_tx
            .get_account(AddressPath::new(Nibbles::unpack(hashed_address)))
            .expect("Failed to get account");
        println!("committed account: {:?}", account);

        let empty_overlay_calc = TrieDbOverlayStateRoot::new(
            provider.tx().expect("Failed to create RO transaction"),
            TrieInput::default(),
        );
        let empty_overlay_state_root = empty_overlay_calc
            .incremental_root()
            .expect("Failed to calculate empty overlay state root");

        assert_eq!(
            overlay_state_root, committed_state_root,
            "State root with storage overlay should match TrieDB after commit"
        );
        assert_eq!(
            committed_state_root, empty_overlay_state_root,
            "TrieDB state root should match empty overlay calculation"
        );
    }
}
