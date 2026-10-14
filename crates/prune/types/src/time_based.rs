/// Time-based retention for distance-pruned bodies, receipts, and account/storage history.
///
/// The configured block distance is replaced by this policy. Other modes and segments are
/// unchanged. Catch-up starts at each segment's actual prune checkpoint, not its configured
/// distance, and advances in historical seconds per second of chain progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(any(test, feature = "serde"), derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(any(test, feature = "serde"), serde(default, deny_unknown_fields))]
pub struct TimeBasedPruning {
    /// Minimum history age to retain, in seconds. Defaults to 31 days. Must be nonzero.
    pub retention_seconds: u64,
    /// Cutoff advancement as a percentage of elapsed chain time. Defaults to 130 (1.3x).
    /// Must be at least 100; 100 maintains the initial window without catching up.
    pub catch_up_percent: u64,
}

impl Default for TimeBasedPruning {
    fn default() -> Self {
        Self { retention_seconds: 31 * 24 * 60 * 60, catch_up_percent: 130 }
    }
}
