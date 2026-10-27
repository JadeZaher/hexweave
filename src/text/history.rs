use super::{TILES, WIDTH};
use std::collections::VecDeque;

pub(super) struct Record {
    pub features: Vec<f32>,
    pub correction: Vec<f32>,
    pub versions: [u64; TILES],
    pub shared_version: u64,
    pub step: usize,
}

#[derive(Default)]
pub(super) struct History {
    records: VecDeque<Record>,
    pub rejected: usize,
    pub peak: usize,
}

impl History {
    pub fn prune(&mut self, versions: &[u64; TILES], shared: u64, step: usize) {
        let before = self.records.len();
        self.records.retain(|record| {
            step.saturating_sub(record.step) <= 8
                && compatible(shared, record.shared_version)
                && versions
                    .iter()
                    .zip(record.versions)
                    .all(|(&now, old)| compatible(now, old))
        });
        self.rejected += before - self.records.len();
    }

    pub fn summary(&self) -> Vec<f32> {
        let mut result = vec![0.0; WIDTH];
        for record in &self.records {
            for (value, correction) in result.iter_mut().zip(&record.correction) {
                *value += correction / self.records.len() as f32;
            }
        }
        result
    }

    pub fn latest(&self) -> Option<&Record> {
        self.records.back()
    }

    pub fn push(&mut self, record: Record) {
        if self.records.len() == 8 {
            self.records.pop_front();
        }
        self.records.push_back(record);
        self.peak = self.peak.max(self.records.len());
    }
}

fn compatible(now: u64, old: u64) -> bool {
    now >= old && now - old <= 1
}
