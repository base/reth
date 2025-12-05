//! TrieDB integration for Reth
//!
//! This crate provides adapters and integrations to use TrieDB as the underlying
//! storage backend for Reth's trie operations, replacing the traditional table-based
//! approach for better performance and native merkle proof generation.

pub mod provider;

pub use provider::TrieDbProviderFactory;
