//! TrieDB provider factory implementation
//!
//! This module provides the `TrieDbProviderFactory` that creates TrieDB-backed
//! providers for use in Reth's multiproof system and sparse trie generation.

use reth_provider::providers::{TrieDbProvider, TrieDbTransaction};
use reth_storage_errors::provider::ProviderResult;

/// Factory for creating TrieDB-backed providers
#[derive(Debug, Clone)]
pub struct TrieDbProviderFactory {
    provider: TrieDbProvider,
}

impl TrieDbProviderFactory {
    /// Create a new TrieDB provider factory
    pub fn new(provider: TrieDbProvider) -> Self {
        Self { provider }
    }

    /// Get a read-only TrieDB provider
    pub fn provider_ro(&self) -> ProviderResult<TrieDbProviderReadOnly> {
        let tx = self.provider.tx()?;
        Ok(TrieDbProviderReadOnly::new(tx))
    }

    /// Get a read-write TrieDB provider  
    pub fn provider_rw(&self) -> ProviderResult<TrieDbProviderReadWrite> {
        let tx = self.provider.tx_mut()?;
        Ok(TrieDbProviderReadWrite::new(tx))
    }
}

/// Read-only TrieDB provider that implements both cursor factory traits
#[derive(Debug)]
pub struct TrieDbProviderReadOnly {
    tx: TrieDbTransaction,
}

impl TrieDbProviderReadOnly {
    /// Create a new read-only provider
    pub fn new(tx: TrieDbTransaction) -> Self {
        Self { tx }
    }

    /// Get the underlying transaction
    pub fn tx(&self) -> &TrieDbTransaction {
        &self.tx
    }

    /// Consume the provider and return the transaction
    pub fn into_tx(self) -> TrieDbTransaction {
        self.tx
    }
}

/// Read-write TrieDB provider that implements both cursor factory traits
#[derive(Debug)]
pub struct TrieDbProviderReadWrite {
    tx: TrieDbTransaction,
}

impl TrieDbProviderReadWrite {
    /// Create a new read-write provider
    pub fn new(tx: TrieDbTransaction) -> Self {
        Self { tx }
    }

    /// Get the underlying transaction
    pub fn tx(&self) -> &TrieDbTransaction {
        &self.tx
    }

    /// Consume the provider and return the transaction
    pub fn into_tx(self) -> TrieDbTransaction {
        self.tx
    }

    /// Commit the transaction
    pub fn commit(self) -> ProviderResult<()> {
        self.tx.commit()
    }
}
