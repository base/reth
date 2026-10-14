use crate::PrunerError;
use alloy_consensus::BlockHeader;
use reth_prune_types::{PruneCheckpoint, PruneSegment, TimeBasedPruning};
use reth_storage_api::{HeaderProvider, MetadataProvider, MetadataWriter};
use tracing::debug;

/// Resolves a time-based cutoff using an anchor committed with the prune checkpoints.
pub(crate) fn prune_target<Provider>(
    provider: &Provider,
    policy: TimeBasedPruning,
    segment: PruneSegment,
    tip: u64,
    previous: Option<PruneCheckpoint>,
    minimum_distance: u64,
) -> Result<Option<u64>, PrunerError>
where
    Provider: HeaderProvider + MetadataProvider + MetadataWriter,
{
    let head_timestamp = timestamp(provider, tip)?;
    let key = format!("pruning.time.v1.{segment}");
    let stored = provider.get_metadata(&key)?;
    let anchor = stored.as_deref().map(Anchor::decode).transpose()?;
    let anchor = match anchor {
        Some(anchor) if anchor.policy == policy => anchor,
        _ => {
            // Changing the policy must not apply a new multiplier to past elapsed time, or
            // suddenly make a shorter retention window eligible. Re-anchor at actual progress.
            let anchor = Anchor {
                head_timestamp,
                cutoff_timestamp: timestamp(
                    provider,
                    previous.and_then(|cp| cp.block_number).unwrap_or_default().min(tip),
                )?,
                policy,
            };
            provider.write_metadata(&key, anchor.encode())?;
            debug!(target: "pruner", ?segment, head_timestamp, cutoff_timestamp = anchor.cutoff_timestamp,
                ?policy, "Initialized time-based pruning anchor");
            anchor
        }
    };

    // Use canonical chain time, never wall time. Saturation pauses advancement during an unwind
    // before the anchor. Whole-second headers conservatively retain the entire boundary second,
    // including on chains with multiple blocks sharing a timestamp.
    let elapsed = head_timestamp.saturating_sub(anchor.head_timestamp);
    let advance = (u128::from(elapsed) * u128::from(policy.catch_up_percent) / 100)
        .min(u128::from(u64::MAX)) as u64;
    let cutoff = head_timestamp
        .saturating_sub(policy.retention_seconds)
        .min(anchor.cutoff_timestamp.saturating_add(advance));
    let Some(max_block) = tip.checked_sub(minimum_distance) else { return Ok(None) };

    // Find the first block whose timestamp is >= cutoff. Headers are retained even when bodies
    // and state history are pruned. Binary search also handles the 2s -> 200ms transition without
    // chain-specific fork configuration.
    let mut low = 0;
    let mut high = max_block;
    if timestamp(provider, high)? < cutoff {
        return Ok(Some(high))
    }
    while low < high {
        let mid = low + (high - low) / 2;
        if timestamp(provider, mid)? < cutoff {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let target = low.checked_sub(1);
    debug!(target: "pruner", ?segment, head_timestamp, cutoff_timestamp = cutoff, ?target,
        target_lag_seconds = head_timestamp.saturating_sub(cutoff).saturating_sub(policy.retention_seconds),
        "Resolved time-based pruning target");
    Ok(target)
}

fn timestamp(provider: &impl HeaderProvider, block: u64) -> Result<u64, PrunerError> {
    Ok(provider
        .header_by_number(block)?
        .ok_or(PrunerError::InconsistentData("missing header for time-based pruning"))?
        .timestamp())
}

/// A fixed-width, versioned metadata value avoids changing the prune checkpoint database codec.
struct Anchor {
    head_timestamp: u64,
    cutoff_timestamp: u64,
    policy: TimeBasedPruning,
}

impl Anchor {
    fn encode(&self) -> Vec<u8> {
        [
            self.head_timestamp,
            self.cutoff_timestamp,
            self.policy.retention_seconds,
            self.policy.catch_up_percent,
        ]
        .into_iter()
        .flat_map(u64::to_be_bytes)
        .collect()
    }

    fn decode(bytes: &[u8]) -> Result<Self, PrunerError> {
        let bytes: &[u8; 32] = bytes.try_into().map_err(|_| {
            PrunerError::InconsistentData("invalid time-based pruning anchor metadata")
        })?;
        let words: [u64; 4] =
            core::array::from_fn(|i| u64::from_be_bytes(bytes.as_chunks::<8>().0[i]));
        Ok(Self {
            head_timestamp: words[0],
            cutoff_timestamp: words[1],
            policy: TimeBasedPruning { retention_seconds: words[2], catch_up_percent: words[3] },
        })
    }
}
