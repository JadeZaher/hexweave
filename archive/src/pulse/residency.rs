use super::math::clipped;
use super::node::{Gradients, Node, ROUTE_FEATURES};
use super::{THRESHOLD, WIDTH};
use crate::ternary::Evidence;
use anyhow::Result;

struct Resident {
    node: usize,
    last_use: u64,
    transform: Evidence,
    activation: Evidence,
    router: Evidence,
}

pub(super) struct Residents {
    entries: Vec<Resident>,
    limit: usize,
    clock: u64,
    pub evictions: usize,
    pub loads: usize,
    pub peak: usize,
    pub peak_bytes: usize,
    pub transform_transitions: usize,
    pub activation_transitions: usize,
    pub router_transitions: usize,
}

impl Residents {
    pub fn new(limit: usize) -> Self {
        Self {
            entries: Vec::with_capacity(limit),
            limit,
            clock: 0,
            evictions: 0,
            loads: 0,
            peak: 0,
            peak_bytes: 0,
            transform_transitions: 0,
            activation_transitions: 0,
            router_transitions: 0,
        }
    }

    pub fn apply(&mut self, id: usize, node: &mut Node, gradients: &Gradients) -> Result<()> {
        let index = self.admit(id, node)?;
        let state = &mut self.entries[index];
        let transform = node.transform.apply_gradient(
            &mut state.transform,
            &clipped(&gradients.transform)?,
            THRESHOLD,
        )?;
        let activation = node.activation.apply_gradient(
            &mut state.activation,
            &clipped(&gradients.activation)?,
            THRESHOLD,
        )?;
        let router = node.router.apply_gradient(
            &mut state.router,
            &clipped(&gradients.router)?,
            THRESHOLD,
        )?;
        self.transform_transitions += transform;
        self.activation_transitions += activation;
        self.router_transitions += router;
        if transform + activation + router > 0 {
            node.version += 1;
        }
        Ok(())
    }

    fn admit(&mut self, id: usize, node: &Node) -> Result<usize> {
        self.clock += 1;
        if let Some(index) = self.entries.iter().position(|entry| entry.node == id) {
            self.entries[index].last_use = self.clock;
            return Ok(index);
        }
        if self.entries.len() == self.limit {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_use)
                .unwrap()
                .0;
            self.entries.swap_remove(oldest);
            self.evictions += 1;
        }
        self.entries.push(Resident {
            node: id,
            last_use: self.clock,
            transform: Evidence::new(WIDTH * WIDTH)?,
            activation: Evidence::new(WIDTH)?,
            router: Evidence::new(node.router.len())?,
        });
        self.loads += 1;
        self.peak = self.peak.max(self.entries.len());
        self.peak_bytes = self.peak_bytes.max(self.current_evidence_bytes());
        Ok(self.entries.len() - 1)
    }

    pub fn bytes_per_resident() -> usize {
        2 * (WIDTH * WIDTH + WIDTH + 7 * ROUTE_FEATURES)
    }

    pub fn current_evidence_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.transform.bytes() + entry.activation.bytes() + entry.router.bytes())
            .sum()
    }
}
