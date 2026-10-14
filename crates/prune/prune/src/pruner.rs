//! Support for pruning.

use crate::{
    segments::{PruneInput, Segment},
    Metrics, PruneLimiter, PrunerError, PrunerEvent,
};
use alloy_primitives::BlockNumber;
use reth_exex_types::FinishedExExHeight;
use reth_primitives_traits::FastInstant as Instant;
use reth_provider::{
    DBProvider, DatabaseProviderFactory, PruneCheckpointReader, PruneCheckpointWriter,
    StageCheckpointReader,
};
use reth_prune_types::{
    PruneProgress, PruneSegment, PrunedSegmentInfo, PrunerOutput, TimeBasedPruning,
};
use reth_stages_types::StageId;
use reth_storage_api::{HeaderProvider, MetadataProvider, MetadataWriter};
use reth_tokio_util::{EventSender, EventStream};
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, instrument};

/// Result of [`Pruner::run`] execution.
pub type PrunerResult = Result<PrunerOutput, PrunerError>;

/// The pruner type itself with the result of [`Pruner::run`]
pub type PrunerWithResult<S, DB> = (Pruner<S, DB>, PrunerResult);

/// Pruner with preset provider factory.
pub type PrunerWithFactory<PF> = Pruner<<PF as DatabaseProviderFactory>::ProviderRW, PF>;

/// Pruning routine. Main pruning logic happens in [`Pruner::run`].
#[derive(Debug)]
pub struct Pruner<Provider, PF> {
    /// Provider factory. If pruner is initialized without it, it will be set to `()`.
    provider_factory: PF,
    segments: Vec<Box<dyn Segment<Provider>>>,
    /// Minimum pruning interval measured in blocks. All prune segments are checked and, if needed,
    /// pruned, when the chain advances by the specified number of blocks.
    min_block_interval: usize,
    /// Previous tip block number when the pruner was run. Even if no data was pruned, this block
    /// number is updated with the tip block number the pruner was called with. It's used in
    /// conjunction with `min_block_interval` to determine when the pruning needs to be initiated.
    previous_tip_block_number: Option<BlockNumber>,
    /// Maximum total entries to prune (delete from database) per run.
    delete_limit: usize,
    /// Maximum time for one pruner run.
    timeout: Option<Duration>,
    /// Optional override for the minimum pruning distance. When set, this replaces the
    /// per-segment hardcoded minimums (e.g. `MINIMUM_UNWIND_SAFE_DISTANCE`).
    minimum_pruning_distance: Option<u64>,
    /// Optional time-based replacement for user history distance modes.
    time_based: Option<TimeBasedPruning>,
    /// The finished height of all `ExEx`'s.
    finished_exex_height: watch::Receiver<FinishedExExHeight>,
    #[doc(hidden)]
    metrics: Metrics,
    event_sender: EventSender<PrunerEvent>,
}

impl<Provider> Pruner<Provider, ()> {
    /// Creates a new [Pruner] without a provider factory.
    pub fn new(
        segments: Vec<Box<dyn Segment<Provider>>>,
        min_block_interval: usize,
        delete_limit: usize,
        timeout: Option<Duration>,
        finished_exex_height: watch::Receiver<FinishedExExHeight>,
    ) -> Self {
        Self {
            provider_factory: (),
            segments,
            min_block_interval,
            previous_tip_block_number: None,
            delete_limit,
            timeout,
            minimum_pruning_distance: None,
            time_based: None,
            finished_exex_height,
            metrics: Metrics::default(),
            event_sender: Default::default(),
        }
    }
}

impl<PF> Pruner<PF::ProviderRW, PF>
where
    PF: DatabaseProviderFactory,
{
    /// Creates a new pruner with the given provider factory.
    pub fn new_with_factory(
        provider_factory: PF,
        segments: Vec<Box<dyn Segment<PF::ProviderRW>>>,
        min_block_interval: usize,
        delete_limit: usize,
        timeout: Option<Duration>,
        finished_exex_height: watch::Receiver<FinishedExExHeight>,
    ) -> Self {
        Self {
            provider_factory,
            segments,
            min_block_interval,
            previous_tip_block_number: None,
            delete_limit,
            timeout,
            minimum_pruning_distance: None,
            time_based: None,
            finished_exex_height,
            metrics: Metrics::default(),
            event_sender: Default::default(),
        }
    }
}

impl<Provider, S> Pruner<Provider, S> {
    /// Sets the time-based retention policy without changing pruning execution limits.
    pub const fn with_time_based(mut self, policy: Option<TimeBasedPruning>) -> Self {
        self.time_based = policy;
        self
    }

    /// Sets the minimum pruning distance, overriding per-segment hardcoded minimums.
    pub const fn with_minimum_pruning_distance(mut self, distance: u64) -> Self {
        self.minimum_pruning_distance = Some(distance);
        self
    }
}

impl<Provider, S> Pruner<Provider, S>
where
    Provider: PruneCheckpointReader
        + PruneCheckpointWriter
        + StageCheckpointReader
        + HeaderProvider
        + MetadataProvider
        + MetadataWriter,
{
    /// Listen for events on the pruner.
    pub fn events(&self) -> EventStream<PrunerEvent> {
        self.event_sender.new_listener()
    }

    /// Run the pruner with the given provider. This will only prune data up to the highest finished
    /// `ExEx` height, if there are no `ExExes`.
    ///
    /// Returns a [`PruneProgress`], indicating whether pruning is finished, or there is more data
    /// to prune.
    #[instrument(
        name = "Pruner::run_with_provider",
        level = "debug",
        target = "pruner",
        skip(self, provider)
    )]
    pub fn run_with_provider(
        &mut self,
        provider: &Provider,
        tip_block_number: BlockNumber,
    ) -> PrunerResult {
        if let Some(policy) = self.time_based &&
            (policy.retention_seconds == 0 || policy.catch_up_percent < 100)
        {
            return Err(PrunerError::InconsistentData(
                "time-based pruning requires positive retention_seconds and catch_up_percent >= 100",
            ))
        }
        let Some(tip_block_number) =
            self.adjust_tip_block_number_to_finished_exex_height(tip_block_number)
        else {
            return Ok(PruneProgress::Finished.into())
        };
        if tip_block_number == 0 {
            self.previous_tip_block_number = Some(tip_block_number);

            debug!(target: "pruner", %tip_block_number, "Nothing to prune yet");
            return Ok(PruneProgress::Finished.into())
        }

        self.event_sender.notify(PrunerEvent::Started { tip_block_number });

        debug!(target: "pruner", %tip_block_number, "Pruner started");
        let start = Instant::now();

        let mut limiter = PruneLimiter::default().set_deleted_entries_limit(self.delete_limit);
        if let Some(timeout) = self.timeout {
            limiter = limiter.set_time_limit(timeout);
        };

        let (stats, deleted_entries, output) =
            self.prune_segments(provider, tip_block_number, &mut limiter)?;

        self.previous_tip_block_number = Some(tip_block_number);

        let elapsed = start.elapsed();
        self.metrics.duration_seconds.record(elapsed);

        output.debug_log(tip_block_number, deleted_entries, elapsed);

        self.event_sender.notify(PrunerEvent::Finished { tip_block_number, elapsed, stats });

        Ok(output)
    }

    /// Prunes the segments that the [Pruner] was initialized with, and the segments that needs to
    /// be pruned according to the highest `static_files`. Segments are parts of the database that
    /// represent one or more tables.
    ///
    /// Returns a list of stats per pruned segment, total number of entries pruned, and
    /// [`PruneProgress`].
    #[instrument(level = "debug", target = "pruner", skip_all, fields(segments = self.segments.len()))]
    fn prune_segments(
        &mut self,
        provider: &Provider,
        tip_block_number: BlockNumber,
        limiter: &mut PruneLimiter,
    ) -> Result<(Vec<PrunedSegmentInfo>, usize, PrunerOutput), PrunerError> {
        let mut stats = Vec::with_capacity(self.segments.len());
        let mut pruned = 0;
        let mut output = PrunerOutput {
            progress: PruneProgress::Finished,
            segments: Vec::with_capacity(self.segments.len()),
        };

        for segment in &self.segments {
            if limiter.is_limit_reached() {
                output.progress =
                    output.progress.combine(PruneProgress::HasMoreData(limiter.interrupt_reason()));
                break
            }

            // Do not initialize time-based anchors until the segment can actually run.
            if let Some(required_stage) = segment.required_stage() &&
                !is_stage_finished(provider, required_stage)?
            {
                debug!(target: "pruner", segment = ?segment.segment(), ?required_stage,
                    "Segment's required stage not finished, skipping");
                continue
            }
            let previous_checkpoint = provider.get_prune_checkpoint(segment.segment())?;
            if let Some((to_block, prune_mode)) = segment
                .mode()
                .map(|mode| {
                    if let Some(policy) = self.time_based &&
                        mode.is_distance() &&
                        segment.purpose().is_user() &&
                        matches!(
                            segment.segment(),
                            PruneSegment::Bodies |
                                PruneSegment::Receipts |
                                PruneSegment::AccountHistory |
                                PruneSegment::StorageHistory
                        )
                    {
                        return crate::time_based::prune_target(
                            provider,
                            policy,
                            segment.segment(),
                            tip_block_number,
                            previous_checkpoint,
                            self.minimum_pruning_distance
                                .unwrap_or_else(|| segment.segment().min_blocks()),
                        )
                        .map(|target| target.map(|block| (block, mode)))
                    }
                    mode.prune_target_block_with_min(
                        tip_block_number,
                        segment.segment(),
                        segment.purpose(),
                        self.minimum_pruning_distance,
                    )
                    .map_err(PrunerError::from)
                })
                .transpose()?
                .flatten()
            {
                debug!(
                    target: "pruner",
                    segment = ?segment.segment(),
                    purpose = ?segment.purpose(),
                    %to_block,
                    ?prune_mode,
                    "Segment pruning started"
                );

                let segment_start = Instant::now();
                let segment_output = segment.prune(
                    provider,
                    PruneInput { previous_checkpoint, to_block, limiter: limiter.clone() },
                )?;
                if let Some(checkpoint) = segment_output.checkpoint {
                    segment
                        .save_checkpoint(provider, checkpoint.as_prune_checkpoint(prune_mode))?;
                }
                self.metrics
                    .get_prune_segment_metrics(segment.segment())
                    .duration_seconds
                    .record(segment_start.elapsed());
                if let Some(highest_pruned_block) =
                    segment_output.checkpoint.and_then(|checkpoint| checkpoint.block_number)
                {
                    self.metrics
                        .get_prune_segment_metrics(segment.segment())
                        .highest_pruned_block
                        .set(highest_pruned_block as f64);
                }

                output.progress = output.progress.combine(segment_output.progress);
                output.segments.push((segment.segment(), segment_output));

                debug!(
                    target: "pruner",
                    segment = ?segment.segment(),
                    purpose = ?segment.purpose(),
                    %to_block,
                    ?prune_mode,
                    %segment_output.pruned,
                    "Segment pruning finished"
                );

                if segment_output.pruned > 0 {
                    limiter.increment_deleted_entries_count_by(segment_output.pruned);
                    pruned += segment_output.pruned;
                    let info = PrunedSegmentInfo {
                        segment: segment.segment(),
                        pruned: segment_output.pruned,
                        progress: segment_output.progress,
                    };
                    stats.push(info);
                }
            } else {
                debug!(target: "pruner", segment = ?segment.segment(), purpose = ?segment.purpose(), "Nothing to prune for the segment");
            }
        }

        Ok((stats, pruned, output))
    }

    /// Returns `true` if the pruning is needed at the provided tip block number.
    /// This is determined by the check against minimum pruning interval and last pruned block
    /// number.
    pub fn is_pruning_needed(&self, tip_block_number: BlockNumber) -> bool {
        let Some(tip_block_number) =
            self.adjust_tip_block_number_to_finished_exex_height(tip_block_number)
        else {
            return false
        };

        // Saturating subtraction is needed for the case when the chain was reverted, meaning
        // current block number might be less than the previous tip block number.
        // If that's the case, no pruning is needed as outdated data is also reverted.
        if tip_block_number.saturating_sub(self.previous_tip_block_number.unwrap_or_default()) >=
            self.min_block_interval as u64
        {
            debug!(
                target: "pruner",
                previous_tip_block_number = ?self.previous_tip_block_number,
                %tip_block_number,
                "Minimum pruning interval reached"
            );
            true
        } else {
            false
        }
    }

    /// Adjusts the tip block number to the finished `ExEx` height. This is needed to not prune more
    /// data than `ExExs` have processed. Depending on the height:
    /// - [`FinishedExExHeight::NoExExs`] returns the tip block number as no adjustment for `ExExs`
    ///   is needed.
    /// - [`FinishedExExHeight::NotReady`] returns `None` as not all `ExExs` have emitted a
    ///   `FinishedHeight` event yet.
    /// - [`FinishedExExHeight::Height`] returns the finished `ExEx` height.
    fn adjust_tip_block_number_to_finished_exex_height(
        &self,
        tip_block_number: BlockNumber,
    ) -> Option<BlockNumber> {
        match *self.finished_exex_height.borrow() {
            FinishedExExHeight::NoExExs => Some(tip_block_number),
            FinishedExExHeight::NotReady => {
                debug!(target: "pruner", %tip_block_number, "Not all ExExs have emitted a `FinishedHeight` event yet, can't prune");
                None
            }
            FinishedExExHeight::Height(finished_exex_height) => {
                debug!(target: "pruner", %tip_block_number, %finished_exex_height, "Adjusting tip block number to the finished ExEx height");
                Some(finished_exex_height)
            }
        }
    }
}

impl<PF> Pruner<PF::ProviderRW, PF>
where
    PF: DatabaseProviderFactory<
        ProviderRW: PruneCheckpointWriter
                        + PruneCheckpointReader
                        + StageCheckpointReader
                        + HeaderProvider
                        + MetadataProvider
                        + MetadataWriter,
    >,
{
    /// Run the pruner. This will only prune data up to the highest finished ExEx height, if there
    /// are no ExExes.
    ///
    /// Returns a [`PruneProgress`], indicating whether pruning is finished, or there is more data
    /// to prune.
    #[instrument(name = "Pruner::run", level = "debug", target = "pruner", skip(self))]
    pub fn run(&mut self, tip_block_number: BlockNumber) -> PrunerResult {
        let provider = self.provider_factory.database_provider_rw()?;
        let result = self.run_with_provider(&provider, tip_block_number);
        provider.commit()?;
        result
    }
}

/// Checks if the given stage has caught up with the `Finish` stage.
///
/// Returns `true` if the stage checkpoint is >= the Finish stage checkpoint.
fn is_stage_finished<Provider: StageCheckpointReader>(
    provider: &Provider,
    stage_id: StageId,
) -> Result<bool, PrunerError> {
    let stage_checkpoint = provider.get_stage_checkpoint(stage_id)?.map(|c| c.block_number);
    let finish_checkpoint = provider.get_stage_checkpoint(StageId::Finish)?.map(|c| c.block_number);

    Ok(stage_checkpoint >= finish_checkpoint)
}

#[cfg(test)]
mod tests {
    use crate::{Pruner, PrunerBuilder};
    use alloy_consensus::Header;
    use alloy_primitives::Address;
    use reth_config::PruneConfig;
    use reth_db_api::{models::StorageSettings, tables};
    use reth_exex_types::FinishedExExHeight;
    use reth_primitives_traits::{Account, SealedHeader, StorageEntry};
    use reth_provider::{test_utils::create_test_provider_factory, PruneCheckpointReader};
    use reth_prune_types::{PruneMode, PruneModes, PruneSegment, TimeBasedPruning};
    use reth_stages::test_utils::TestStageDB;
    use reth_storage_api::StorageSettingsCache;

    #[test]
    fn time_based_catch_up_across_block_time_change_and_restart() {
        let db = TestStageDB::default();
        db.factory.set_storage_settings_cache(StorageSettings::v1());
        let headers = (0..=700)
            .map(|number| {
                SealedHeader::seal_slow(Header {
                    number,
                    timestamp: if number <= 100 { number * 2 } else { 200 + (number - 100) / 5 },
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();
        db.insert_headers(&headers).unwrap();
        let changesets = (0..=700)
            .map(|_| vec![(Address::ZERO, Account::default(), vec![StorageEntry::default()])])
            .collect::<Vec<_>>();
        db.insert_changesets(changesets.clone(), None).unwrap();
        db.insert_history(changesets, None).unwrap();

        // Establish different actual checkpoints with the original block-distance policy.
        let mut config = PruneConfig {
            minimum_pruning_distance: 5,
            segments: PruneModes {
                account_history: Some(PruneMode::Distance(100)),
                storage_history: Some(PruneMode::Distance(110)),
                ..Default::default()
            },
            ..Default::default()
        };
        PrunerBuilder::new(config.clone())
            .build_with_provider_factory(db.factory.clone())
            .run(150)
            .unwrap();

        // The nominal distance would retain everything in this small chain. Time retention
        // must replace it, without jumping either segment straight to the desired cutoff.
        config.segments.account_history = Some(PruneMode::Distance(13_392_000));
        config.segments.storage_history = Some(PruneMode::Distance(13_392_000));
        config.segments.time_based =
            Some(TimeBasedPruning { retention_seconds: 100, catch_up_percent: 130 });
        let (exex_tx, exex_rx) = tokio::sync::watch::channel(FinishedExExHeight::NoExExs);

        // An interrupted first run must not leave an anchor ahead of committed progress.
        {
            let provider = db.factory.provider_rw().unwrap();
            PrunerBuilder::new(config.clone())
                .build_with_provider_factory(db.factory.clone())
                .run_with_provider(&provider, 100)
                .unwrap();
            // Dropping the provider aborts this transaction, including anchor initialization.
        }

        // At head timestamp 220, the cutoffs are 113 and 93: six old blocks per segment,
        // not fifty. Later, both converge to 100 seconds and cross into repeated timestamps.
        for (tip, account_cutoff, storage_cutoff) in
            [(150, 50, 40), (200, 56, 46), (350, 74, 65), (650, 149, 149), (700, 199, 199)]
        {
            // Recreate the pruner every time; catch-up must survive process restarts.
            let mut pruner = PrunerBuilder::new(config.clone())
                .finished_exex_height(exex_rx.clone())
                .build_with_provider_factory(db.factory.clone());
            pruner.run(tip).unwrap();
            let provider = db.factory.provider().unwrap();
            for (segment, cutoff) in [
                (PruneSegment::AccountHistory, account_cutoff),
                (PruneSegment::StorageHistory, storage_cutoff),
            ] {
                assert_eq!(
                    provider.get_prune_checkpoint(segment).unwrap().unwrap().block_number,
                    Some(cutoff)
                );
            }
            assert_eq!(
                db.table::<tables::AccountChangeSets>()
                    .unwrap()
                    .into_iter()
                    .map(|(block, _)| block)
                    .collect::<Vec<_>>(),
                ((account_cutoff + 1)..=700).collect::<Vec<_>>()
            );
            assert_eq!(
                db.table::<tables::StorageChangeSets>()
                    .unwrap()
                    .into_iter()
                    .map(|(key, _)| key.block_number())
                    .collect::<Vec<_>>(),
                ((storage_cutoff + 1)..=700).collect::<Vec<_>>()
            );
            assert_eq!(
                db.table::<tables::AccountsHistory>().unwrap()[0].1.iter().collect::<Vec<_>>(),
                ((account_cutoff + 1)..=700).collect::<Vec<_>>()
            );
            assert_eq!(
                db.table::<tables::StoragesHistory>().unwrap()[0].1.iter().collect::<Vec<_>>(),
                ((storage_cutoff + 1)..=700).collect::<Vec<_>>()
            );
        }

        // A shorter window and faster multiplier must re-anchor, not accelerate retroactively.
        config.segments.time_based =
            Some(TimeBasedPruning { retention_seconds: 50, catch_up_percent: 200 });
        let mut pruner =
            PrunerBuilder::new(config.clone()).build_with_provider_factory(db.factory.clone());
        pruner.run(700).unwrap();
        assert_eq!(db.table::<tables::AccountChangeSets>().unwrap()[0].0, 200);

        // A lower head (unwind) or lagging ExEx cannot advance pruning.
        pruner.run(650).unwrap();
        exex_tx.send(FinishedExExHeight::Height(650)).unwrap();
        let mut pruner = PrunerBuilder::new(config.clone())
            .finished_exex_height(exex_rx)
            .build_with_provider_factory(db.factory.clone());
        pruner.run(700).unwrap();
        assert_eq!(db.table::<tables::AccountChangeSets>().unwrap()[0].0, 200);

        // Time-based eligibility still cannot cross the configured block safety floor.
        let headers = (701..=750)
            .map(|number| {
                SealedHeader::seal_slow(Header {
                    number,
                    timestamp: 200 + (number - 100) / 5,
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();
        db.insert_headers(&headers).unwrap();
        config.minimum_pruning_distance = 530;
        PrunerBuilder::new(config.clone())
            .build_with_provider_factory(db.factory.clone())
            .run(750)
            .unwrap();
        assert_eq!(db.table::<tables::AccountChangeSets>().unwrap()[0].0, 221);

        config.segments.time_based.as_mut().unwrap().retention_seconds = 0;
        assert!(PrunerBuilder::new(config)
            .build_with_provider_factory(db.factory.clone())
            .run(750)
            .is_err());
        assert_eq!(db.table::<tables::AccountChangeSets>().unwrap()[0].0, 221);
    }

    #[test]
    fn is_pruning_needed() {
        let provider_factory = create_test_provider_factory();

        let (finished_exex_height_tx, finished_exex_height_rx) =
            tokio::sync::watch::channel(FinishedExExHeight::NoExExs);

        let mut pruner =
            Pruner::new_with_factory(provider_factory, vec![], 5, 0, None, finished_exex_height_rx);

        // No last pruned block number was set before
        let first_block_number = 1;
        assert!(!pruner.is_pruning_needed(first_block_number));
        pruner.previous_tip_block_number = Some(first_block_number);

        // Tip block number delta is >= than min block interval
        let second_block_number = first_block_number + pruner.min_block_interval as u64;
        assert!(pruner.is_pruning_needed(second_block_number));
        pruner.previous_tip_block_number = Some(second_block_number);

        // Tip block number delta is < than min block interval
        assert!(!pruner.is_pruning_needed(second_block_number));

        // Tip block number delta is >= than min block interval
        let third_block_number = second_block_number + pruner.min_block_interval as u64;
        assert!(pruner.is_pruning_needed(third_block_number));

        // Not all ExExs have emitted a `FinishedHeight` event yet
        finished_exex_height_tx.send(FinishedExExHeight::NotReady).unwrap();
        assert!(!pruner.is_pruning_needed(third_block_number));

        // Adjust tip block number to the finished ExEx height that doesn't reach the threshold
        finished_exex_height_tx.send(FinishedExExHeight::Height(second_block_number)).unwrap();
        assert!(!pruner.is_pruning_needed(third_block_number));

        // Adjust tip block number to the finished ExEx height that reaches the threshold
        finished_exex_height_tx.send(FinishedExExHeight::Height(third_block_number)).unwrap();
        assert!(pruner.is_pruning_needed(third_block_number));
    }
}
