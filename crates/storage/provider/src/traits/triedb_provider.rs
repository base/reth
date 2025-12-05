#![allow(missing_docs)]

use crate::providers::{TrieDbProvider, TrieDbTransaction};

/// TrieDB provider factory.
pub trait TrieDbProviderFactory {
    /// Create new instance of TrieDB provider.
    fn triedb_provider(&self) -> TrieDbProvider;
}

pub trait TrieDbTxProvider {
    fn triedb_tx_ref(&self) -> &TrieDbTransaction;
    fn triedb_tx(&mut self) -> &mut TrieDbTransaction;
    fn into_triedb_tx(self) -> TrieDbTransaction;
}
