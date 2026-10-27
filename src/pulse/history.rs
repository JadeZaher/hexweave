use super::state::Model;
use super::{FEATURES, TILES, WIDTH};
use std::collections::VecDeque;
use std::mem::size_of;

pub(super) const HISTORY_CAPACITY: usize = 8;
const MAX_AGE: usize = 8;
const MAX_VERSION_LAG: u64 = 1;

#[derive(Clone)]
pub(super) struct Record {
    pub node: usize,
    pub node_version: u64,
    pub shared_version: u64,
    pub dependency_versions: [u64; TILES],
    pub step: usize,
    pub features: Vec<f32>,
    pub correction: Vec<f32>,
}

pub(super) struct History {
    records: VecDeque<Record>,
    pub incompatible_drops: usize,
    pub capacity_drops: usize,
    pub peak: usize,
}

impl History {
    pub fn new() -> Self {
        Self {
            records: VecDeque::with_capacity(HISTORY_CAPACITY),
            incompatible_drops: 0,
            capacity_drops: 0,
            peak: 0,
        }
    }

    pub fn prune(&mut self, model: &Model, step: usize) {
        let before = self.records.len();
        self.records.retain(|record| {
            step.saturating_sub(record.step) <= MAX_AGE
                && model.nodes[record.node]
                    .version
                    .saturating_sub(record.node_version)
                    <= MAX_VERSION_LAG
                && model.shared_version.saturating_sub(record.shared_version) <= MAX_VERSION_LAG
                && model
                    .nodes
                    .iter()
                    .zip(record.dependency_versions)
                    .all(|(node, version)| node.version.saturating_sub(version) <= MAX_VERSION_LAG)
        });
        self.incompatible_drops += before - self.records.len();
    }

    pub fn push(&mut self, record: Record) {
        if self.records.len() == HISTORY_CAPACITY {
            self.records.pop_front();
            self.capacity_drops += 1;
        }
        self.records.push_back(record);
        self.peak = self.peak.max(self.records.len());
    }

    pub fn latest(&self) -> Option<Record> {
        self.records.back().cloned()
    }

    pub fn summary(&self) -> Vec<f32> {
        let mut result = vec![0.0; WIDTH];
        if !self.records.is_empty() {
            for record in &self.records {
                for (entry, value) in result.iter_mut().zip(&record.correction) {
                    *entry += value / self.records.len() as f32;
                }
            }
        }
        result
    }

    pub fn record_payload_bytes() -> usize {
        (FEATURES + WIDTH) * size_of::<f32>()
            + 2 * size_of::<usize>()
            + (2 + TILES) * size_of::<u64>()
    }
}
