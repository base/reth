use super::{
    progress::{
        ArchiveDownloadProgress, DownloadProgress, DownloadRequestLimiter, DownloadRequestPermit,
        RequestPriority, SharedProgress, SharedProgressWriter,
    },
    session::DownloadSession,
    RETRY_BACKOFF_SECS,
};
use eyre::Result;
use reqwest::{blocking::Client as BlockingClient, header::RANGE, StatusCode};
use reth_cli_util::cancellation::CancellationToken;
use reth_fs_util as fs;
use std::{
    any::Any,
    collections::{BTreeMap, VecDeque},
    fs::{File, OpenOptions},
    io::{self, BufWriter, Read, Write},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Condvar, Mutex, MutexGuard,
    },
    thread,
    time::Duration,
};
use tracing::{info, warn};
use url::Url;

/// Maximum retry attempts for a single download segment.
const SEGMENT_RETRY_ATTEMPTS: u32 = 3;

/// Minimum archive size that benefits from segmented downloads.
const SEGMENTED_DOWNLOAD_MIN_FILE_SIZE: u64 = 128 * 1024 * 1024;

/// Piece sizes are large so big downloads do not create too many requests while
/// still giving multiple workers enough work to do.
const SEGMENTED_DOWNLOAD_SMALL_PIECE_SIZE: u64 = 32 * 1024 * 1024;
const SEGMENTED_DOWNLOAD_LARGE_PIECE_SIZE: u64 = 64 * 1024 * 1024;

/// Cap exponential piece retry backoff to avoid overly long stalls.
const SEGMENTED_DOWNLOAD_MAX_BACKOFF_SECS: u64 = 30;

/// Segmented piece requests should time out quickly enough to recover from slow or stalled
/// requests.
const SEGMENTED_DOWNLOAD_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Bytes a piece download buffers before each write to the shared `.part` file.
///
/// Filesystems such as ext4 take an exclusive per-inode lock for every buffered write, so many
/// pieces writing small chunks into one file contend on that lock instead of the disk.
const SEGMENTED_DOWNLOAD_WRITE_BUFFER_SIZE: usize = 8 * 1024 * 1024;

/// Paths for one downloaded archive and its `.part` file.
#[derive(Debug, Clone)]
struct DownloadPaths {
    /// User-facing archive file name derived from the URL.
    file_name: String,
    /// Final path for the completed archive file.
    final_path: PathBuf,
    /// Temporary path used while the archive is still downloading.
    part_path: PathBuf,
}

impl DownloadPaths {
    /// Builds the final and partial download paths from the archive URL.
    fn from_url(url: &str, target_dir: &Path) -> Self {
        let file_name = Url::parse(url)
            .ok()
            .and_then(|u| u.path_segments()?.next_back().map(|s| s.to_string()))
            .unwrap_or_else(|| "snapshot.tar".to_string());

        Self {
            final_path: target_dir.join(&file_name),
            part_path: target_dir.join(format!("{file_name}.part")),
            file_name,
        }
    }

    /// Returns the user-facing file name derived from the archive URL.
    fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Returns the final on-disk path for the completed archive.
    fn final_path(&self) -> &Path {
        &self.final_path
    }

    /// Returns the partial download path used while the archive is still in flight.
    fn part_path(&self) -> &Path {
        &self.part_path
    }

    /// Promotes the partial file into the final archive path.
    fn finalize(&self) -> Result<()> {
        fs::rename(&self.part_path, &self.final_path)?;
        Ok(())
    }

    /// Removes only the partial `.part` file for the current archive.
    fn cleanup_partial(&self) {
        let _ = fs::remove_file(&self.part_path);
    }

    /// Removes both final and partial archive files so a fresh attempt can restart cleanly.
    fn cleanup_all(&self) {
        let _ = fs::remove_file(&self.final_path);
        self.cleanup_partial();
    }
}

/// Fetches one archive to disk and chooses sequential or segmented download.
pub(crate) struct ArchiveFetcher {
    /// Remote archive URL.
    url: String,
    /// On-disk paths used for this archive download.
    paths: DownloadPaths,
    /// Shared command-scoped download state.
    session: DownloadSession,
}

impl ArchiveFetcher {
    /// Creates a fetcher for one archive URL under the given target directory.
    pub(crate) fn new(url: impl Into<String>, target_dir: &Path, session: DownloadSession) -> Self {
        let url = url.into();
        let paths = DownloadPaths::from_url(&url, target_dir);
        Self { url, paths, session }
    }

    /// Downloads the archive using the best strategy supported by the remote source.
    ///
    /// When `availability` is set, the downloaded bytes are published to it so the archive can
    /// be read while it downloads. Segmented downloads publish the contiguous prefix as pieces
    /// land; every other path publishes the whole archive once it is on disk.
    pub(crate) fn download(
        &self,
        download_progress: Option<&mut ArchiveDownloadProgress<'_>>,
        availability: Option<&ArchiveAvailability>,
    ) -> Result<DownloadedArchive> {
        let downloaded = self.download_archive(download_progress, availability)?;
        if let Some(availability) = availability {
            availability.complete(&downloaded)?;
        }
        Ok(downloaded)
    }

    /// Downloads the archive, publishing progress to `availability` only on the segmented path.
    fn download_archive(
        &self,
        download_progress: Option<&mut ArchiveDownloadProgress<'_>>,
        availability: Option<&ArchiveAvailability>,
    ) -> Result<DownloadedArchive> {
        if let Some(path) = archive_file_url_path(&self.url)? {
            let size = fs::metadata(&path)?.len();
            if !self.quiet() {
                info!(target: "reth::cli",
                    file = %path.display(),
                    size = %DownloadProgress::format_size(size),
                    "Using local archive"
                );
            }
            return Ok(DownloadedArchive { path, size })
        }

        let Some(request_limiter) = self.session.request_limiter() else {
            return self.download_sequential(super::MAX_DOWNLOAD_RETRIES, download_progress)
        };

        let probe = self.probe(request_limiter)?;

        match choose_fetch_strategy(probe) {
            FetchStrategy::Sequential(reason) => {
                self.log_sequential_fallback(reason, probe.total_size);
                self.download_sequential(super::MAX_DOWNLOAD_RETRIES, download_progress)
            }
            FetchStrategy::Segmented(plan) => {
                self.download_segmented(probe.total_size, plan, download_progress, availability)
            }
        }
    }

    /// Removes any archive files created by this fetcher.
    pub(crate) fn cleanup_downloaded_files(&self) {
        self.paths.cleanup_all();
    }

    /// Probes the remote source for file size and HTTP range support.
    fn probe(&self, request_limiter: &DownloadRequestLimiter) -> Result<RemoteArchiveProbe> {
        // Probes are single-byte requests that let an archive start, so they go first.
        let permit = request_limiter.acquire(
            self.session.progress(),
            self.session.cancel_token(),
            RequestPriority::new(u64::MAX, None),
        )?;
        let client = permit.client()?;
        let mut attempt = 1;
        let probe = loop {
            let probe = client
                .get(&self.url)
                .header(RANGE, "bytes=0-0")
                .send()
                .and_then(|response| response.error_for_status());
            // A transient failure here would permanently downgrade the archive to one stream.
            if probe.is_ok() || attempt == SEGMENT_RETRY_ATTEMPTS {
                break probe
            }
            std::thread::sleep(self.session.retry_delay(piece_retry_backoff(attempt, false)));
            attempt += 1;
        };

        let (supports_ranges, total_size) = match probe {
            Ok(response) if response.status() == StatusCode::PARTIAL_CONTENT => {
                let total = response
                    .headers()
                    .get("Content-Range")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.split('/').next_back())
                    .and_then(|value| value.parse::<u64>().ok());
                (true, total)
            }
            _ => {
                let response = client.head(&self.url).send()?.error_for_status()?;
                (false, response.content_length())
            }
        };

        Ok(RemoteArchiveProbe {
            total_size: total_size.ok_or_else(|| eyre::eyre!("Server did not return file size"))?,
            supports_ranges,
        })
    }

    /// Downloads the archive as a single resumable stream using one request at a time.
    fn download_sequential(
        &self,
        max_download_retries: u32,
        mut download_progress: Option<&mut ArchiveDownloadProgress<'_>>,
    ) -> Result<DownloadedArchive> {
        let quiet = self.quiet();

        if !quiet {
            info!(target: "reth::cli", file = %self.paths.file_name(), "Connecting to download server");
        }

        let mut total_size: Option<u64> = None;
        let mut last_error: Option<eyre::Error> = None;

        for attempt in 1..=max_download_retries {
            let existing_size =
                fs::metadata(self.paths.part_path()).map(|meta| meta.len()).unwrap_or(0);

            if let Some(total) = total_size &&
                existing_size >= total
            {
                return self.finalize_download(total)
            }

            if attempt > 1 {
                info!(target: "reth::cli",
                    file = %self.paths.file_name(),
                    "Retry attempt {}/{} - resuming from {} bytes",
                    attempt, max_download_retries, existing_size
                );
            }

            // Sequential downloads are smaller than the segmented minimum unless the server lacks
            // range support, so they rank behind larger archives.
            let request_permit = self
                .session
                .request_limiter()
                .map(|limiter| {
                    limiter.acquire(
                        self.session.progress(),
                        self.session.cancel_token(),
                        RequestPriority::new(0, None),
                    )
                })
                .transpose()?;
            let fallback_client;
            let client = match &request_permit {
                Some(permit) => permit.client()?,
                None => {
                    fallback_client = BlockingClient::builder().http1_only().build()?;
                    &fallback_client
                }
            };

            let mut request = client.get(&self.url).timeout(Duration::from_secs(30));
            if existing_size > 0 {
                request = request.header(RANGE, format!("bytes={existing_size}-"));
                if !quiet && attempt == 1 {
                    info!(target: "reth::cli", file = %self.paths.file_name(), "Resuming from {} bytes", existing_size);
                }
            }

            let response = match request.send().and_then(|response| response.error_for_status()) {
                Ok(response) => response,
                Err(error) => {
                    last_error = Some(error.into());
                    if attempt < max_download_retries {
                        info!(target: "reth::cli",
                            file = %self.paths.file_name(),
                            retry_delay = ?self.session.retry_delay(Duration::from_secs(RETRY_BACKOFF_SECS)),
                            "Download failed, retrying"
                        );
                        std::thread::sleep(
                            self.session.retry_delay(Duration::from_secs(RETRY_BACKOFF_SECS)),
                        );
                    }
                    continue;
                }
            };

            let is_partial = response.status() == StatusCode::PARTIAL_CONTENT;
            let size = if is_partial {
                response
                    .headers()
                    .get("Content-Range")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.split('/').next_back())
                    .and_then(|value| value.parse().ok())
            } else {
                response.content_length()
            };

            if total_size.is_none() {
                total_size = size;
                if !quiet && let Some(size) = size {
                    info!(target: "reth::cli",
                        file = %self.paths.file_name(),
                        size = %DownloadProgress::format_size(size),
                        "Downloading"
                    );
                }
            }

            let current_total = total_size.ok_or_else(|| {
                eyre::eyre!("Server did not provide Content-Length or Content-Range header")
            })?;

            let file = if is_partial && existing_size > 0 {
                OpenOptions::new()
                    .append(true)
                    .open(self.paths.part_path())
                    .map_err(|error| fs::FsPathError::open(error, self.paths.part_path()))?
            } else {
                fs::create_file(self.paths.part_path())?
            };

            let start_offset = if is_partial { existing_size } else { 0 };
            let mut reader = response;

            let copy_result;
            let flush_result;

            if let Some(progress) = self.session.progress() {
                let mut on_written = |bytes| {
                    if let Some(download_progress) = download_progress.as_deref_mut() {
                        download_progress.record_downloaded(bytes);
                    }
                };
                let mut writer = SharedProgressWriter {
                    inner: BufWriter::new(file),
                    progress: Arc::clone(progress),
                    on_written: Some(&mut on_written),
                };
                copy_result = io::copy(&mut reader, &mut writer);
                flush_result = writer.inner.flush();
            } else {
                let mut progress = DownloadProgress::new(current_total);
                progress.downloaded = start_offset;
                let mut writer = ProgressWriter {
                    inner: BufWriter::new(file),
                    progress,
                    cancel_token: self.session.cancel_token().clone(),
                };
                copy_result = io::copy(&mut reader, &mut writer);
                flush_result = writer.inner.flush();
                println!();
            }

            if let Err(error) = copy_result.and(flush_result) {
                last_error = Some(error.into());
                if attempt < max_download_retries {
                    info!(target: "reth::cli",
                        file = %self.paths.file_name(),
                        retry_delay = ?self.session.retry_delay(Duration::from_secs(RETRY_BACKOFF_SECS)),
                        "Download interrupted, retrying"
                    );
                    std::thread::sleep(
                        self.session.retry_delay(Duration::from_secs(RETRY_BACKOFF_SECS)),
                    );
                }
                continue;
            }

            return self.finalize_download(current_total)
        }

        Err(last_error.unwrap_or_else(|| {
            eyre::eyre!("Download failed after {} attempts", max_download_retries)
        }))
    }

    /// Downloads the archive by splitting it into large range-request pieces.
    fn download_segmented(
        &self,
        total_size: u64,
        plan: SegmentedDownloadPlan,
        download_progress: Option<&mut ArchiveDownloadProgress<'_>>,
        availability: Option<&ArchiveAvailability>,
    ) -> Result<DownloadedArchive> {
        let request_limiter = self.session.require_request_limiter()?;
        info!(target: "reth::cli",
            total_size = %DownloadProgress::format_size(total_size),
            piece_size = %DownloadProgress::format_size(plan.piece_size),
            pieces = plan.piece_count,
            max_concurrent_requests = request_limiter.max_concurrency(),
            "Starting queued segmented download"
        );

        SegmentedDownload::new(
            self.url.clone(),
            self.paths.clone(),
            total_size,
            plan,
            self.session.clone(),
            download_progress,
        )
        .run(availability)
    }

    /// Logs why this archive must fall back to the sequential fetch path.
    fn log_sequential_fallback(&self, reason: SequentialDownloadFallback, total_size: u64) {
        match reason {
            SequentialDownloadFallback::NoRangeSupport => {
                info!(target: "reth::cli",
                    file = %self.paths.file_name(),
                    "Server does not support Range requests, falling back to sequential download"
                );
            }
            SequentialDownloadFallback::EmptyFile => {
                info!(target: "reth::cli",
                    file = %self.paths.file_name(),
                    "Remote archive is empty, falling back to sequential download"
                );
            }
            SequentialDownloadFallback::TooSmall => {
                let _ = total_size;
            }
        }
    }

    /// Finalizes the downloaded archive and returns its on-disk location and size.
    fn finalize_download(&self, size: u64) -> Result<DownloadedArchive> {
        self.paths.finalize()?;
        if !self.quiet() {
            info!(target: "reth::cli", file = %self.paths.file_name(), "Download complete");
        }
        Ok(DownloadedArchive { path: self.paths.final_path().to_path_buf(), size })
    }

    /// Returns `true` when this fetch should stay quiet because shared progress is active.
    fn quiet(&self) -> bool {
        self.session.progress().is_some()
    }
}

/// Resolves a `file://` archive URL to its local path.
fn archive_file_url_path(url: &str) -> Result<Option<PathBuf>> {
    let Ok(parsed) = Url::parse(url) else { return Ok(None) };
    if parsed.scheme() != "file" {
        return Ok(None)
    }

    parsed
        .to_file_path()
        .map(Some)
        .map_err(|_| eyre::eyre!("Invalid file:// archive URL path: {url}"))
}

/// The final path and size of one archive fetched to disk.
#[derive(Debug, Clone)]
pub(crate) struct DownloadedArchive {
    /// Final on-disk path for the downloaded archive.
    pub(crate) path: PathBuf,
    /// Total archive size in bytes.
    pub(crate) size: u64,
}

/// The prefix of an archive that a fetch has written to disk, so extraction can read the
/// archive while it is still downloading.
#[derive(Default)]
pub(crate) struct ArchiveAvailability {
    /// Current fetch state.
    state: Mutex<AvailabilityState>,
    /// Wakes readers when the state changes.
    changed: Condvar,
    /// Archive bytes the reader has consumed.
    consumed: AtomicU64,
}

/// Fetch state shared between one archive download and its reader.
#[derive(Default)]
enum AvailabilityState {
    /// The fetch has not opened the archive file yet.
    #[default]
    Pending,
    /// The first `available` of `total` bytes of `file` are written.
    Open {
        /// Archive file, shared with readers so a later rename or delete does not affect them.
        file: Arc<File>,
        /// Length of the written prefix.
        available: u64,
        /// Total archive size.
        total: u64,
    },
    /// No more bytes will arrive, because the fetch or the reader failed.
    Aborted,
}

impl ArchiveAvailability {
    /// Shares the archive file once it is created, with nothing written yet.
    fn open(&self, file: File, total: u64) {
        let mut state = self.lock();
        if !matches!(*state, AvailabilityState::Aborted) {
            *state = AvailabilityState::Open { file: Arc::new(file), available: 0, total };
            self.changed.notify_all();
        }
    }

    /// Publishes that the first `len` bytes are written. The prefix never shrinks.
    fn publish(&self, len: u64) {
        let mut state = self.lock();
        if let AvailabilityState::Open { available, total, .. } = &mut *state &&
            len > *available
        {
            *available = len.min(*total);
            self.changed.notify_all();
        }
    }

    /// Publishes the whole downloaded archive, opening it if the fetch path never did.
    fn complete(&self, downloaded: &DownloadedArchive) -> Result<()> {
        if matches!(*self.lock(), AvailabilityState::Pending) {
            self.open(fs::open(&downloaded.path)?, downloaded.size);
        }
        self.publish(downloaded.size);
        Ok(())
    }

    /// Fails readers and stops the fetch because the archive will not complete.
    ///
    /// Returns `true` if this call aborted the archive, and `false` if it was already aborted or
    /// fully published.
    pub(crate) fn abort(&self) -> bool {
        let mut state = self.lock();
        match *state {
            AvailabilityState::Aborted => false,
            AvailabilityState::Open { available, total, .. } if available == total => false,
            _ => {
                *state = AvailabilityState::Aborted;
                self.changed.notify_all();
                true
            }
        }
    }

    /// Returns a guard that aborts this availability when dropped, so a reader never blocks
    /// after the fetch returns early or unwinds.
    pub(crate) fn abort_on_drop(&self) -> ArchiveAvailabilityAbortGuard<'_> {
        ArchiveAvailabilityAbortGuard(self)
    }

    /// Returns how far the published prefix is ahead of the reader, in bytes.
    pub(crate) fn extraction_lead(&self) -> u64 {
        let available = match *self.lock() {
            AvailabilityState::Open { available, .. } => available,
            AvailabilityState::Pending | AvailabilityState::Aborted => 0,
        };
        available.saturating_sub(self.consumed.load(Ordering::Relaxed))
    }

    /// Returns whether the archive was aborted.
    fn is_aborted(&self) -> bool {
        matches!(*self.lock(), AvailabilityState::Aborted)
    }

    /// Blocks until bytes past `position` are written, then returns the file and the written
    /// length, which equals `position` only at the end of the archive.
    fn wait_past(&self, position: u64) -> io::Result<(Arc<File>, u64)> {
        let mut state = self.lock();
        loop {
            match &*state {
                AvailabilityState::Aborted => {
                    return Err(io::Error::other("archive download stopped before completing"))
                }
                AvailabilityState::Open { file, available, total }
                    if *available > position || available == total =>
                {
                    return Ok((Arc::clone(file), *available))
                }
                _ => state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner()),
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, AvailabilityState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Aborts an [`ArchiveAvailability`] when dropped.
pub(crate) struct ArchiveAvailabilityAbortGuard<'a>(&'a ArchiveAvailability);

impl Drop for ArchiveAvailabilityAbortGuard<'_> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Reads the published prefix of an archive, blocking at its end until more is written.
pub(crate) struct ArchiveAvailabilityReader<'a> {
    /// Availability published by the fetch.
    availability: &'a ArchiveAvailability,
    /// Next archive offset to read.
    position: u64,
}

impl<'a> ArchiveAvailabilityReader<'a> {
    /// Creates a reader starting at the beginning of the archive.
    pub(crate) const fn new(availability: &'a ArchiveAvailability) -> Self {
        Self { availability, position: 0 }
    }
}

impl Read for ArchiveAvailabilityReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0)
        }

        let (file, available) = self.availability.wait_past(self.position)?;
        let readable = usize::try_from(available - self.position).unwrap_or(usize::MAX);
        if readable == 0 {
            return Ok(0)
        }

        let len = buf.len().min(readable);
        let read = file.read_at(&mut buf[..len], self.position)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "archive is shorter than its published length",
            ))
        }
        self.position += read as u64;
        self.availability.consumed.store(self.position, Ordering::Relaxed);
        Ok(read)
    }
}

/// Remote metadata used to choose between sequential and segmented download.
#[derive(Debug, Clone, Copy)]
struct RemoteArchiveProbe {
    /// Total archive size reported by the remote source.
    total_size: u64,
    /// Whether the remote source supports byte-range requests.
    supports_ranges: bool,
}

/// Reasons the fetcher may choose the sequential download path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequentialDownloadFallback {
    /// The remote source does not support byte-range requests.
    NoRangeSupport,
    /// The remote source reported an empty archive.
    EmptyFile,
    /// The archive is too small to benefit from segmented download.
    TooSmall,
}

/// The fetch strategy chosen after probing the remote source.
#[derive(Debug)]
enum FetchStrategy {
    /// Use the single-stream download path.
    Sequential(SequentialDownloadFallback),
    /// Use the segmented download path.
    Segmented(SegmentedDownloadPlan),
}

/// Chooses the fetch strategy from the remote probe and available worker budget.
fn choose_fetch_strategy(probe: RemoteArchiveProbe) -> FetchStrategy {
    if !probe.supports_ranges {
        return FetchStrategy::Sequential(SequentialDownloadFallback::NoRangeSupport)
    }

    if probe.total_size == 0 {
        return FetchStrategy::Sequential(SequentialDownloadFallback::EmptyFile)
    }

    plan_segmented_download(probe.total_size)
        .map(FetchStrategy::Segmented)
        .unwrap_or(FetchStrategy::Sequential(SequentialDownloadFallback::TooSmall))
}

/// Wrapper that tracks download progress while writing data.
/// Used with [`io::copy`] to display progress during downloads.
struct ProgressWriter<W> {
    /// Wrapped writer receiving downloaded bytes.
    inner: W,
    /// Per-download progress tracker for the legacy path.
    progress: DownloadProgress,
    /// Cancellation token checked between writes.
    cancel_token: CancellationToken,
}

impl<W: Write> Write for ProgressWriter<W> {
    /// Writes bytes, checks cancellation, and updates local download progress.
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.cancel_token.is_cancelled() {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "download cancelled"));
        }
        let n = self.inner.write(buf)?;
        let _ = self.progress.update(n as u64);
        Ok(n)
    }

    /// Flushes the wrapped writer.
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// One queued byte range for a segmented archive download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DownloadPiece {
    /// Inclusive start byte for this piece.
    start: u64,
    /// Inclusive end byte for this piece.
    end: u64,
}

/// Fixed plan for a segmented archive: piece size and piece count.
#[derive(Debug)]
struct SegmentedDownloadPlan {
    /// Bytes assigned to each piece, except possibly the last.
    piece_size: u64,
    /// Number of pieces created for this archive.
    piece_count: usize,
    /// Queue of pieces to download.
    pieces: VecDeque<DownloadPiece>,
}

/// Dispatches one archive's pieces to the shared request slots and retries failed pieces.
struct SegmentedDownload {
    /// Remote archive URL.
    url: String,
    /// On-disk paths used for this archive download.
    paths: DownloadPaths,
    /// Total archive size in bytes.
    total_size: u64,
    /// Piece plan for this archive.
    plan: SegmentedDownloadPlan,
    /// Shared command-scoped download state.
    session: DownloadSession,
}

/// Shared inputs each piece download needs.
#[derive(Clone, Copy)]
struct SegmentedWorkerContext<'a> {
    /// Remote archive URL.
    url: &'a str,
    /// Partial file where pieces are written.
    file: &'a File,
    /// Shared progress counters for the whole command, when enabled.
    shared: Option<&'a Arc<SharedProgress>>,
    /// Cancellation token shared by the whole command.
    cancel_token: &'a CancellationToken,
    /// Session carrying the retry-delay override.
    session: &'a DownloadSession,
    /// Receives the contiguous downloaded prefix, when the archive is read while downloading.
    availability: Option<&'a ArchiveAvailability>,
    /// Piece queue and completion state of this archive.
    state: &'a SegmentedDownloadState,
    /// First fatal piece error.
    terminal_failure: &'a TerminalFailure,
    /// Logical bytes of completed pieces, for shared progress.
    piece_progress_bytes: &'a AtomicU64,
}

impl SegmentedDownload {
    /// Creates the segmented download state for one archive.
    fn new(
        url: String,
        paths: DownloadPaths,
        total_size: u64,
        plan: SegmentedDownloadPlan,
        session: DownloadSession,
        _download_progress: Option<&mut ArchiveDownloadProgress<'_>>,
    ) -> Self {
        Self { url, paths, total_size, plan, session }
    }

    /// Runs the segmented download to completion or returns the first fatal error.
    ///
    /// The calling thread waits for a request slot for one piece at a time, in file order, and
    /// downloads each piece on its own thread while it holds the slot. The number of piece
    /// threads and connections is therefore bounded by the shared request limit, not by the
    /// number of archives. Publishes the contiguous downloaded prefix to `availability` as
    /// pieces land, and stops early once `availability` is aborted.
    fn run(self, availability: Option<&ArchiveAvailability>) -> Result<DownloadedArchive> {
        let Self { url, paths, total_size, plan, session } = self;
        {
            let file = fs::create_file(paths.part_path())?;
            file.set_len(total_size)?;
        }
        if let Some(availability) = availability {
            availability.open(fs::open(paths.part_path())?, total_size);
        }

        let file = OpenOptions::new()
            .write(true)
            .open(paths.part_path())
            .map_err(|error| fs::FsPathError::open(error, paths.part_path()))?;
        let state = SegmentedDownloadState::new(plan.pieces);
        let terminal_failure = TerminalFailure::default();
        let piece_progress_bytes = AtomicU64::new(0);
        let request_limiter = session.require_request_limiter()?;
        let shared = session.progress();
        let cancel_token = session.cancel_token();
        let priority = RequestPriority::new(total_size, availability);
        let context = SegmentedWorkerContext {
            url: &url,
            file: &file,
            shared,
            cancel_token,
            session: &session,
            availability,
            state: &state,
            terminal_failure: &terminal_failure,
            piece_progress_bytes: &piece_progress_bytes,
        };

        std::thread::scope(|scope| {
            let mut handles = Vec::new();

            while let Some(in_flight) = state.next_piece(cancel_token, availability) {
                let permit = match request_limiter.acquire(shared, cancel_token, priority) {
                    Ok(permit) => permit,
                    Err(error) => {
                        state.note_terminal_failure();
                        terminal_failure.record(error);
                        break
                    }
                };
                handles.push(scope.spawn(move || Self::download_piece(context, in_flight, permit)));
            }

            for handle in handles {
                if let Err(payload) = handle.join() {
                    state.note_terminal_failure();
                    terminal_failure.record(eyre::eyre!(
                        "Segmented download worker panicked: {}",
                        panic_payload_message(payload)
                    ));
                }
            }
        });

        let error = terminal_failure.take().or_else(|| {
            availability
                .is_some_and(ArchiveAvailability::is_aborted)
                .then(|| eyre::eyre!("Archive reader stopped before the download completed"))
        });
        if let Some(error) = error {
            if let Some(shared) = shared {
                shared.sub_active_download_bytes(piece_progress_bytes.load(Ordering::Relaxed));
            }
            paths.cleanup_partial();
            return Err(error.wrap_err("Parallel download failed"))
        }

        if let Some(shared) = shared {
            shared.sub_active_download_bytes(piece_progress_bytes.load(Ordering::Relaxed));
            shared.record_archive_download_complete(total_size);
        }

        paths.finalize()?;
        info!(target: "reth::cli", file = %paths.file_name(), "Download complete");
        Ok(DownloadedArchive { path: paths.final_path().to_path_buf(), size: total_size })
    }

    /// Downloads one piece with the request slot `permit`, then records the outcome.
    fn download_piece(
        context: SegmentedWorkerContext<'_>,
        in_flight: InFlightPiece<'_>,
        permit: DownloadRequestPermit<'_>,
    ) {
        let piece = in_flight.piece;
        let result = permit.client().map_err(PieceAttemptFailure::Terminal).and_then(|client| {
            Self::download_piece_with_retries(
                client,
                context.url,
                context.file,
                piece,
                context.shared,
                context.piece_progress_bytes,
                context.cancel_token,
                context.session,
            )
        });
        drop(permit);

        match result {
            Ok(()) => {
                let prefix = context.state.complete(piece);
                if let Some(availability) = context.availability {
                    availability.publish(prefix);
                }
            }
            Err(PieceAttemptFailure::Retryable { error, .. }) if context.state.requeue(piece) => {
                warn!(target: "reth::cli",
                    start = piece.start,
                    end = piece.end,
                    %error,
                    "Piece download failed, requeued"
                );
            }
            Err(
                PieceAttemptFailure::Retryable { error, .. } | PieceAttemptFailure::Terminal(error),
            ) => {
                context.state.note_terminal_failure();
                context.terminal_failure.record(error);
            }
        }
    }

    /// Downloads one queued piece with per-piece retry/backoff.
    ///
    /// The caller holds a slot of the shared request limit for every attempt and backoff, so
    /// whole-file and piece downloads use the same fixed number of HTTP request slots.
    #[expect(clippy::too_many_arguments)]
    fn download_piece_with_retries(
        client: &BlockingClient,
        url: &str,
        file: &File,
        piece: DownloadPiece,
        shared: Option<&Arc<SharedProgress>>,
        piece_progress_bytes: &AtomicU64,
        cancel_token: &CancellationToken,
        session: &DownloadSession,
    ) -> std::result::Result<(), PieceAttemptFailure> {
        for attempt in 1..=SEGMENT_RETRY_ATTEMPTS {
            if cancel_token.is_cancelled() {
                return Err(PieceAttemptFailure::Terminal(eyre::eyre!("Download cancelled")))
            }

            match Self::download_piece_once(
                client,
                url,
                file,
                piece,
                shared,
                piece_progress_bytes,
                cancel_token,
            ) {
                Ok(()) => return Ok(()),
                Err(PieceAttemptFailure::Retryable { error: _, throttled })
                    if attempt < SEGMENT_RETRY_ATTEMPTS =>
                {
                    std::thread::sleep(
                        session.retry_delay(piece_retry_backoff(attempt, throttled)),
                    );
                }
                Err(failure) => return Err(failure),
            }
        }

        Err(PieceAttemptFailure::Terminal(eyre::eyre!(
            "Piece download failed after {SEGMENT_RETRY_ATTEMPTS} attempts"
        )))
    }

    /// Downloads one queued piece once.
    fn download_piece_once(
        client: &BlockingClient,
        url: &str,
        file: &std::fs::File,
        piece: DownloadPiece,
        shared: Option<&Arc<SharedProgress>>,
        piece_progress_bytes: &AtomicU64,
        cancel_token: &CancellationToken,
    ) -> std::result::Result<(), PieceAttemptFailure> {
        let expected_len = piece.end - piece.start + 1;

        let response = match client
            .get(url)
            .header(RANGE, format!("bytes={}-{}", piece.start, piece.end))
            .timeout(Duration::from_secs(SEGMENTED_DOWNLOAD_REQUEST_TIMEOUT_SECS))
            .send()
        {
            Ok(response) if response.status() == StatusCode::PARTIAL_CONTENT => response,
            Ok(response) if should_retry_piece_status(response.status()) => {
                return Err(PieceAttemptFailure::Retryable {
                    error: eyre::eyre!(
                        "Server returned {} for piece {}-{}",
                        response.status(),
                        piece.start,
                        piece.end
                    ),
                    throttled: is_throttle_piece_status(response.status()),
                });
            }
            Ok(response) => {
                return Err(PieceAttemptFailure::Terminal(eyre::eyre!(
                    "Server returned {} instead of 206 for Range request",
                    response.status()
                )));
            }
            Err(error) => {
                return Err(PieceAttemptFailure::Retryable {
                    throttled: is_throttle_piece_error(&error),
                    error: error.into(),
                });
            }
        };

        let mut buf = vec![0u8; SEGMENTED_DOWNLOAD_WRITE_BUFFER_SIZE];
        let mut filled = 0;
        let mut reader = response.take(expected_len);
        // File offset of `buf[0]`.
        let mut offset = piece.start;

        loop {
            if cancel_token.is_cancelled() {
                return Err(PieceAttemptFailure::Terminal(eyre::eyre!("Download cancelled")));
            }

            match reader.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => {
                    filled += n;
                    if filled == buf.len() {
                        file.write_all_at(&buf, offset)
                            .map_err(|error| PieceAttemptFailure::Terminal(error.into()))?;
                        offset += filled as u64;
                        filled = 0;
                    }
                    if let Some(progress) = shared {
                        progress.record_session_fetched_bytes(n as u64);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(PieceAttemptFailure::Retryable {
                        throttled: error.kind() == io::ErrorKind::TimedOut,
                        error: error.into(),
                    });
                }
            }
        }
        file.write_all_at(&buf[..filled], offset)
            .map_err(|error| PieceAttemptFailure::Terminal(error.into()))?;
        offset += filled as u64;

        let downloaded_len = offset - piece.start;
        if downloaded_len == expected_len {
            if let Some(progress) = shared {
                progress.add_active_download_bytes(expected_len);
            }
            piece_progress_bytes.fetch_add(expected_len, Ordering::Relaxed);
            return Ok(())
        }

        Err(PieceAttemptFailure::Retryable {
            error: eyre::eyre!(
                "Piece {}-{} ended early: expected {} bytes, downloaded {}",
                piece.start,
                piece.end,
                expected_len,
                downloaded_len
            ),
            throttled: false,
        })
    }
}

/// Shared queue state for one segmented archive download.
///
/// The dispatcher takes pieces until the queue is empty and no piece can be requeued, or one
/// piece fails the whole attempt.
struct SegmentedDownloadState {
    /// Pieces waiting to be downloaded and pieces in flight.
    queue: Mutex<PieceQueue>,
    /// Wakes the dispatcher when a piece finishes or the attempt fails.
    changed: Condvar,
    /// Set once a piece hits a fatal error.
    failed: AtomicBool,
    /// Remaining number of times a piece that exhausted its retries may be requeued.
    requeues_left: AtomicU64,
    /// Downloaded pieces and the contiguous prefix they form.
    completed: Mutex<CompletedPieces>,
}

/// Pieces of one segmented archive that are not downloaded yet.
struct PieceQueue {
    /// Pieces waiting to be downloaded, in file order except for requeued pieces at the front.
    pieces: VecDeque<DownloadPiece>,
    /// Pieces taken from the queue whose download has not finished.
    in_flight: usize,
}

/// A piece taken from the queue; dropping it marks the piece as no longer in flight.
struct InFlightPiece<'a> {
    /// Queue the piece was taken from.
    state: &'a SegmentedDownloadState,
    /// The piece to download.
    piece: DownloadPiece,
}

impl Drop for InFlightPiece<'_> {
    fn drop(&mut self) {
        // A panicking piece download can no longer complete its piece.
        if thread::panicking() {
            self.state.note_terminal_failure();
        }
        self.state.queue.lock().unwrap().in_flight -= 1;
        self.state.changed.notify_all();
    }
}

/// Downloaded pieces of one segmented archive.
#[derive(Default)]
struct CompletedPieces {
    /// Length of the archive prefix made only of downloaded pieces.
    prefix: u64,
    /// Exclusive end byte of each downloaded piece past `prefix`, keyed by start byte.
    pending: BTreeMap<u64, u64>,
}

impl SegmentedDownloadState {
    /// Creates the shared queue state for one segmented archive attempt.
    fn new(pieces: VecDeque<DownloadPiece>) -> Self {
        let requeues_left = AtomicU64::new(pieces.len() as u64);
        Self {
            queue: Mutex::new(PieceQueue { pieces, in_flight: 0 }),
            changed: Condvar::new(),
            failed: AtomicBool::new(false),
            requeues_left,
            completed: Mutex::default(),
        }
    }

    /// Puts a piece back on the queue, returning `false` once the requeue budget is spent.
    ///
    /// The piece goes to the front so the contiguous prefix that extraction reads is not held
    /// back until the end of the archive.
    fn requeue(&self, piece: DownloadPiece) -> bool {
        if self
            .requeues_left
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| left.checked_sub(1))
            .is_err()
        {
            return false
        }
        self.queue.lock().unwrap().pieces.push_front(piece);
        true
    }

    /// Records a downloaded piece and returns the length of the contiguous downloaded prefix.
    fn complete(&self, piece: DownloadPiece) -> u64 {
        let mut guard = self.completed.lock().unwrap();
        let completed = &mut *guard;
        completed.pending.insert(piece.start, piece.end + 1);
        while let Some(end) = completed.pending.remove(&completed.prefix) {
            completed.prefix = end;
        }
        completed.prefix
    }

    /// Takes the next piece, waiting while the queue is empty but an in-flight piece may still
    /// be requeued.
    ///
    /// Returns `None` once every piece is downloaded, or when cancellation, a fatal error, or
    /// an aborted `availability` stopped the attempt.
    fn next_piece(
        &self,
        cancel_token: &CancellationToken,
        availability: Option<&ArchiveAvailability>,
    ) -> Option<InFlightPiece<'_>> {
        let mut queue = self.queue.lock().unwrap();
        loop {
            if cancel_token.is_cancelled() ||
                self.failed.load(Ordering::Relaxed) ||
                availability.is_some_and(ArchiveAvailability::is_aborted)
            {
                return None
            }
            if let Some(piece) = queue.pieces.pop_front() {
                queue.in_flight += 1;
                return Some(InFlightPiece { state: self, piece })
            }
            if queue.in_flight == 0 {
                return None
            }
            // Wake periodically so cancellation and aborts are observed without a piece
            // finishing.
            queue = self.changed.wait_timeout(queue, Duration::from_millis(100)).unwrap().0;
        }
    }

    /// Marks the entire segmented attempt as failed so the dispatcher stops taking more pieces.
    fn note_terminal_failure(&self) {
        self.failed.store(true, Ordering::Relaxed);
        self.changed.notify_all();
    }
}

/// Stores the first fatal error seen across segmented download workers.
#[derive(Default)]
struct TerminalFailure {
    /// First fatal worker error, if any.
    error: Mutex<Option<eyre::Error>>,
}

impl TerminalFailure {
    /// Stores the first fatal error and ignores later ones from other workers.
    fn record(&self, error: eyre::Error) {
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    /// Returns the stored fatal error after worker execution finishes.
    fn take(&self) -> Option<eyre::Error> {
        self.error.lock().unwrap().take()
    }
}

/// Splits an archive into contiguous byte ranges for segmented download.
fn build_download_pieces(total_size: u64, piece_size: u64) -> VecDeque<DownloadPiece> {
    let mut pieces = VecDeque::new();
    let mut start = 0;

    while start < total_size {
        let end = (start + piece_size).min(total_size) - 1;
        pieces.push_back(DownloadPiece { start, end });
        start = end + 1;
    }

    pieces
}

/// Chooses the fixed piece size for a large archive.
///
/// Smaller large files use 32 MiB pieces so there are enough pieces for several workers.
/// Very large files use 64 MiB pieces to keep the request count down.
fn segmented_piece_size(total_size: u64) -> u64 {
    if total_size < 2 * 1024 * 1024 * 1024 {
        SEGMENTED_DOWNLOAD_SMALL_PIECE_SIZE
    } else {
        SEGMENTED_DOWNLOAD_LARGE_PIECE_SIZE
    }
}

/// Builds the segmented download plan for one archive.
///
/// Small files stay single-stream. Larger files are split into fixed pieces and
/// can use up to the shared request limit.
fn plan_segmented_download(total_size: u64) -> Option<SegmentedDownloadPlan> {
    if total_size < SEGMENTED_DOWNLOAD_MIN_FILE_SIZE {
        return None;
    }

    let piece_size = segmented_piece_size(total_size);
    if total_size <= piece_size {
        return None;
    }

    let pieces = build_download_pieces(total_size, piece_size);
    let piece_count = pieces.len();

    Some(SegmentedDownloadPlan { piece_size, piece_count, pieces })
}

/// Returns the retry backoff for one piece attempt.
fn piece_retry_backoff(attempt: u32, throttled: bool) -> Duration {
    let base = if throttled { 2 } else { RETRY_BACKOFF_SECS };
    let multiplier = 1u64 << attempt.saturating_sub(1).min(3);
    Duration::from_secs(base.saturating_mul(multiplier).min(SEGMENTED_DOWNLOAD_MAX_BACKOFF_SECS))
}

/// Returns whether an HTTP status should retry the current piece.
fn is_retryable_piece_status(status: StatusCode) -> bool {
    matches!(status, StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS) ||
        status.is_server_error()
}

/// Returns whether a piece request should retry after the given status.
fn should_retry_piece_status(status: StatusCode) -> bool {
    status == StatusCode::OK || is_retryable_piece_status(status)
}

/// Returns whether an HTTP status looks like throttling or timeout.
fn is_throttle_piece_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT |
            StatusCode::TOO_MANY_REQUESTS |
            StatusCode::SERVICE_UNAVAILABLE |
            StatusCode::GATEWAY_TIMEOUT
    )
}

/// Returns whether a reqwest error looks like throttling or timeout.
fn is_throttle_piece_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || matches!(error.status(), Some(status) if is_throttle_piece_status(status))
}

/// The result of one piece download attempt.
enum PieceAttemptFailure {
    /// The piece can be retried.
    Retryable { error: eyre::Error, throttled: bool },
    /// The piece failed in a way that should stop the archive.
    Terminal(eyre::Error),
}

/// Converts a thread panic payload into a readable message.
fn panic_payload_message(payload: Box<dyn Any + Send + 'static>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use reth_cli_util::cancellation::CancellationToken;
    use std::io::Write;

    #[test]
    fn segmented_plan_skips_small_files() {
        assert!(plan_segmented_download(SEGMENTED_DOWNLOAD_MIN_FILE_SIZE - 1).is_none());
    }

    #[test]
    fn segmented_plan_uses_small_pieces_below_two_gib() {
        let total_size = 512 * 1024 * 1024;
        let plan = plan_segmented_download(total_size).unwrap();

        assert_eq!(plan.piece_size, SEGMENTED_DOWNLOAD_SMALL_PIECE_SIZE);
        assert_eq!(plan.piece_count, 16);
    }

    #[test]
    fn build_download_pieces_covers_entire_file() {
        let pieces = build_download_pieces(10, 4).into_iter().collect::<Vec<_>>();

        assert_eq!(
            pieces,
            vec![
                DownloadPiece { start: 0, end: 3 },
                DownloadPiece { start: 4, end: 7 },
                DownloadPiece { start: 8, end: 9 },
            ]
        );
    }

    #[test]
    fn piece_status_retry_policy_retries_200_ok() {
        assert!(should_retry_piece_status(StatusCode::OK));
        assert!(should_retry_piece_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(should_retry_piece_status(StatusCode::from_u16(520).unwrap()));
        assert!(!should_retry_piece_status(StatusCode::NOT_FOUND));
    }

    #[test]
    fn choose_fetch_strategy_uses_segmented_when_ranges_are_supported() {
        let strategy = choose_fetch_strategy(RemoteArchiveProbe {
            total_size: 512 * 1024 * 1024,
            supports_ranges: true,
        });

        assert!(matches!(strategy, FetchStrategy::Segmented(_)));
    }

    #[test]
    fn choose_fetch_strategy_falls_back_without_ranges() {
        let strategy = choose_fetch_strategy(RemoteArchiveProbe {
            total_size: 512 * 1024 * 1024,
            supports_ranges: false,
        });

        assert!(matches!(
            strategy,
            FetchStrategy::Sequential(SequentialDownloadFallback::NoRangeSupport)
        ));
    }

    #[test]
    fn archive_fetcher_uses_file_url_archive_directly() {
        let dir = tempfile::tempdir().unwrap();
        let archive_path = dir.path().join("state.tar.zst");
        {
            let mut archive = std::fs::File::create(&archive_path).unwrap();
            archive.write_all(b"local archive bytes").unwrap();
        }

        let cache_dir = dir.path().join("cache");
        std::fs::create_dir(&cache_dir).unwrap();
        let url = Url::from_file_path(&archive_path).unwrap().to_string();
        let session = DownloadSession::new(None, None, CancellationToken::new());
        let fetcher = ArchiveFetcher::new(url, &cache_dir, session);

        let downloaded = fetcher.download(None, None).unwrap();

        assert_eq!(downloaded.path, archive_path);
        assert_eq!(downloaded.size, b"local archive bytes".len() as u64);
        assert!(!cache_dir.join("state.tar.zst").exists());
        assert!(!cache_dir.join("state.tar.zst.part").exists());
    }

    #[test]
    fn segmented_state_requeues_failed_pieces_within_budget() {
        let state = SegmentedDownloadState::new(build_download_pieces(8, 4));
        let token = CancellationToken::new();
        let first = state.next_piece(&token, None).unwrap().piece;

        assert!(state.requeue(first));
        assert!(state.requeue(first));
        assert!(!state.requeue(first));
        assert_eq!(state.queue.lock().unwrap().pieces.front(), Some(&first));
    }

    #[test]
    fn segmented_state_prefix_stops_at_first_missing_piece() {
        let pieces = build_download_pieces(10, 4).into_iter().collect::<Vec<_>>();
        let state = SegmentedDownloadState::new(VecDeque::new());

        assert_eq!(state.complete(pieces[2]), 0);
        assert_eq!(state.complete(pieces[0]), 4);
        assert_eq!(state.complete(pieces[1]), 10);
    }

    #[test]
    fn availability_reader_follows_published_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("archive.part");
        let data = (0..=255u8).cycle().take(1000).collect::<Vec<_>>();
        std::fs::write(&path, &data).unwrap();
        let availability = ArchiveAvailability::default();

        let read = std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let mut read = Vec::new();
                ArchiveAvailabilityReader::new(&availability).read_to_end(&mut read).map(|_| read)
            });
            availability.open(fs::open(&path).unwrap(), data.len() as u64);
            for len in [300, 299, 1000] {
                availability.publish(len);
            }
            reader.join().unwrap()
        });

        assert_eq!(read.unwrap(), data);
        assert!(!availability.abort(), "a fully published archive stays readable");
    }

    #[test]
    fn availability_reader_fails_when_download_aborts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("archive.part");
        std::fs::write(&path, [7u8; 100]).unwrap();
        let availability = ArchiveAvailability::default();
        availability.open(fs::open(&path).unwrap(), 100);
        availability.publish(40);

        let mut reader = ArchiveAvailabilityReader::new(&availability);
        let mut buf = [0u8; 100];
        assert_eq!(reader.read(&mut buf).unwrap(), 40);
        assert!(availability.abort());

        let error = reader.read(&mut buf).unwrap_err();
        assert_eq!(error.to_string(), "archive download stopped before completing");
    }

    #[test]
    fn request_slots_feed_a_large_archive_extractor_ahead_of_small_archives() {
        const GIB: u64 = 1 << 30;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("archive.part");
        std::fs::write(&path, []).unwrap();
        let large = ArchiveAvailability::default();
        large.open(fs::open(&path).unwrap(), 700 * GIB);
        let small = ArchiveAvailability::default();
        small.open(fs::open(&path).unwrap(), GIB);

        // Returns the order in which small archives and then the large archive, queued behind
        // them, get the only slot.
        let serve_order = || {
            let limiter = DownloadRequestLimiter::new(1);
            let token = CancellationToken::new();
            let order = Mutex::new(Vec::new());
            std::thread::scope(|scope| {
                let held = limiter.acquire(None, &token, RequestPriority::new(0, None)).unwrap();
                let queue = |name, priority, waiting| {
                    let (limiter, token, order) = (&limiter, &token, &order);
                    scope.spawn(move || {
                        let _permit = limiter.acquire(None, token, priority).unwrap();
                        order.lock().unwrap().push(name);
                    });
                    while limiter.waiting_requests() < waiting {
                        std::thread::yield_now();
                    }
                };
                for waiting in 1..=3 {
                    queue("small", RequestPriority::new(GIB, Some(&small)), waiting);
                }
                queue("large", RequestPriority::new(700 * GIB, Some(&large)), 4);
                drop(held);
            });
            order.into_inner().unwrap()
        };

        assert_eq!(serve_order(), ["large", "small", "small", "small"]);

        // Once its extractor has a full window to read, the large archive yields.
        large.publish(9 * GIB);
        assert_eq!(serve_order(), ["small", "small", "small", "large"]);
    }
}
