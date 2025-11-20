use crate::ProviderError;
use triedb::transaction::TransactionError;

impl From<TransactionError> for ProviderError {
    fn from(err: TransactionError) -> Self {
        TrieDBError::TransactionError(err.to_string()).into()
    }
}

impl From<triedb::database::Error> for ProviderError {
    fn from(err: triedb::database::Error) -> Self {
        TrieDBError::DatabaseError(format!("{:?}", err)).into()
    }
}

impl From<triedb::database::OpenError> for ProviderError {
    fn from(err: triedb::database::OpenError) -> Self {
        TrieDBError::OpenError(format!("{:?}", err)).into()
    }
}

/// TrieDB error type.
#[derive(Clone, Debug, thiserror::Error)]
pub enum TrieDBError {
    /// Error opening TrieDB
    #[error("received triedb database open error: {_0}")]
    OpenError(String),
    /// Error opening a read-only or read-write transaction from TrieDB
    #[error("received triedb database error: {_0}")]
    DatabaseError(String),
    /// Error opening a read-only or read-write transaction from TrieDB
    #[error("received triedb transaction error: {_0}")]
    TransactionError(String),
    /// Attempting to write using a read-only transaction
    #[error("received reth triedb read-only write error")]
    ReadOnlyWriteError,
    /// An assertion check failed
    #[error("triedb assertion failed: {_0}")]
    AssertionError(String),
}
