//! Thread-local cumulative monotonic timers for persistence overhead; see AGENTS.md §ledger.

use std::cell::Cell;
use std::time::Instant;

/// Timed regions; buckets may nest (Scan inside StoreWrite), so totals are not additive.
#[derive(Clone, Copy, Debug)]
pub(super) enum Bucket {
    StoreRead,
    StoreWrite,
    Scan,
    Publish,
    Sync,
    SourceVerify,
    SourceGuard,
}

pub(super) const BUCKETS: usize = 7;

/// Cumulative nanoseconds and calls per bucket.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Totals {
    pub nanos: [u64; BUCKETS],
    pub calls: [u64; BUCKETS],
}

thread_local! {
    static TOTALS: Cell<Totals> = Cell::new(Totals::default());
}

/// Run `work`, charging its monotonic wall time to `bucket` on this thread.
pub(super) fn time<T>(bucket: Bucket, work: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = work();
    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    TOTALS.with(|cell| {
        let mut totals = cell.get();
        let index = bucket as usize;
        totals.nanos[index] = totals.nanos[index].saturating_add(elapsed);
        totals.calls[index] = totals.calls[index].saturating_add(1);
        cell.set(totals);
    });
    result
}

/// Snapshot of this thread's cumulative totals.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn snapshot() -> Totals {
    TOTALS.with(Cell::get)
}

impl Totals {
    /// Per-bucket difference `self - earlier`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn since(&self, earlier: &Totals) -> Totals {
        Totals {
            nanos: std::array::from_fn(|i| self.nanos[i].saturating_sub(earlier.nanos[i])),
            calls: std::array::from_fn(|i| self.calls[i].saturating_sub(earlier.calls[i])),
        }
    }
}
