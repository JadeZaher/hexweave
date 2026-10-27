use crate::{config::Config, topology::Mesh};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Action {
    Stop,
    Single(usize),
    Fork(usize, usize),
}

#[derive(Clone, Debug, Serialize)]
pub struct Branch {
    pub tile: usize,
    pub depth: usize,
    pub history: Vec<usize>,
    pub mass: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Visit {
    pub branch: Branch,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ExitReason {
    Requested,
    DepthLimit,
    VisitLimit,
    FrontierLimit,
    NoNeighbors,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exit {
    pub visit: usize,
    pub weight: f64,
    pub reason: ExitReason,
}

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub visits: Vec<Visit>,
    pub exits: Vec<Exit>,
    pub peak_frontier: usize,
}

/// Plan bounded branch metadata only; see src/AGENTS.md, "Planner".
pub fn plan<F>(config: &Config, mesh: &Mesh, start: usize, mut choose: F) -> Result<Plan>
where
    F: FnMut(&Branch, &[usize]) -> Result<Action>,
{
    config.validate()?;
    ensure!(
        mesh.len() == config.tile_count()?,
        "mesh and configuration disagree"
    );
    ensure!(start < mesh.len(), "start tile is outside the mesh");
    let mut plan = Plan {
        visits: vec![Visit {
            branch: Branch {
                tile: start,
                depth: 0,
                history: vec![start],
                mass: 1.0,
            },
            parent: None,
            children: Vec::new(),
        }],
        exits: Vec::new(),
        peak_frontier: 1,
    };
    let mut pending = VecDeque::from([0]);
    while let Some(id) = pending.pop_front() {
        let branch = &plan.visits[id].branch;
        let neighbors = mesh.neighbors(branch.tile)?;
        let forced = if branch.depth + 1 >= config.max_depth {
            Some(ExitReason::DepthLimit)
        } else if neighbors.is_empty() {
            Some(ExitReason::NoNeighbors)
        } else {
            None
        };
        if let Some(reason) = forced {
            finish(&mut plan, id, reason);
            continue;
        }
        let destinations = destinations(choose(branch, &neighbors)?, &neighbors, config.fanout)?;
        let reason = if destinations.is_empty() {
            Some(ExitReason::Requested)
        } else if plan.visits.len() + destinations.len() > config.max_visits {
            Some(ExitReason::VisitLimit)
        } else if pending.len() + destinations.len() > config.max_frontier {
            Some(ExitReason::FrontierLimit)
        } else {
            None
        };
        if let Some(reason) = reason {
            finish(&mut plan, id, reason);
        } else {
            extend(&mut plan, &mut pending, id, &destinations);
        }
    }
    Ok(plan)
}

fn destinations(action: Action, legal: &[usize], fanout: bool) -> Result<Vec<usize>> {
    let destinations = match action {
        Action::Stop => Vec::new(),
        Action::Single(tile) => vec![tile],
        Action::Fork(left, right) => {
            ensure!(fanout, "fork requested while fanout is disabled");
            ensure!(left != right, "fork destinations must be distinct");
            vec![left, right]
        }
    };
    for tile in &destinations {
        ensure!(
            legal.contains(tile),
            "destination {tile} is not a neighboring tile"
        );
    }
    Ok(destinations)
}

fn finish(plan: &mut Plan, id: usize, reason: ExitReason) {
    plan.exits.push(Exit {
        visit: id,
        weight: plan.visits[id].branch.mass,
        reason,
    });
}

fn extend(plan: &mut Plan, pending: &mut VecDeque<usize>, parent: usize, targets: &[usize]) {
    let branch = plan.visits[parent].branch.clone();
    for tile in targets {
        let mut history = branch.history.clone();
        history.push(*tile);
        let id = plan.visits.len();
        plan.visits.push(Visit {
            branch: Branch {
                tile: *tile,
                depth: branch.depth + 1,
                history,
                mass: branch.mass / targets.len() as f64,
            },
            parent: Some(parent),
            children: Vec::new(),
        });
        plan.visits[parent].children.push(id);
        pending.push_back(id);
    }
    plan.peak_frontier = plan.peak_frontier.max(pending.len());
}
