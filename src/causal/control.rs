use super::checkpoint_owner::{OwnerEnvelope, Stop, Terminal};
use super::child_protocol::{Request, StopFile};
use super::{child_io, launch::OwnerProcess};
use anyhow::{ensure, Context, Result};
use std::time::Instant;

pub(super) struct Control<'a> {
    pub request: &'a Request,
    pub started: Instant,
    pub observer_sequence: u64,
    pub stopped: Option<Terminal>,
    process: OwnerProcess,
}

pub(super) fn millis(started: Instant) -> Result<u64> {
    Ok(u64::try_from(started.elapsed().as_millis())?)
}

impl<'a> Control<'a> {
    pub fn new(request: &'a Request, process: OwnerProcess, started: Instant) -> Self {
        Self {
            request,
            started,
            process,
            observer_sequence: request.owner.observer_sequence,
            stopped: None,
        }
    }

    pub fn poll(&mut self) -> Result<Option<Terminal>> {
        self.process.alive()?;
        if self.request.stop_path.try_exists()? {
            let stop: StopFile =
                serde_json::from_slice(&child_io::read(&self.request.stop_path, 2048)?)?;
            ensure!(
                stop.schema == 1 && stop.observer_sequence >= self.observer_sequence,
                "invalid STOP observer sequence"
            );
            ensure!(
                matches!(
                    stop.reason,
                    Terminal::UserStop | Terminal::ResourceStop | Terminal::BudgetStop
                ),
                "owner failure/invalid STOP is terminal without resumable success"
            );
            if let Some(old) = self.stopped {
                ensure!(old == stop.reason, "STOP category changed");
            }
            self.stopped = Some(stop.reason);
            self.observer_sequence = stop.observer_sequence;
        }
        if self.stopped.is_none()
            && millis(self.started)?
                .checked_add(self.request.limits.stop_reserve_millis)
                .context("deadline overflow")?
                >= self.request.limits.wall_millis
        {
            self.stopped = Some(Terminal::BudgetStop);
        }
        Ok(self.stopped)
    }

    pub fn envelope(
        &self,
        physical_opportunities: u64,
        terminal: Option<Terminal>,
    ) -> Result<OwnerEnvelope> {
        let mut owner = self.request.owner.clone();
        owner.consumed.opportunities = owner
            .consumed
            .opportunities
            .checked_add(physical_opportunities)
            .context("owner opportunity overflow")?;
        owner.consumed.wall_millis = owner
            .consumed
            .wall_millis
            .checked_add(millis(self.started)?)
            .context("owner wall overflow")?;
        owner.observer_sequence = self.observer_sequence;
        owner.stop = match terminal.or(self.stopped) {
            Some(Terminal::UserStop) => Stop::UserStop,
            Some(Terminal::ResourceStop) => Stop::ResourceStop,
            Some(Terminal::BudgetStop) => Stop::BudgetStop,
            Some(Terminal::Failed) => Stop::Failed,
            Some(Terminal::Completed) | None => Stop::Running,
        };
        Ok(owner)
    }

    pub fn terminal(&mut self, physical_opportunities: u64) -> Result<Terminal> {
        if let Some(stop) = self.poll()? {
            return Ok(stop);
        }
        let total = self
            .request
            .owner
            .consumed
            .opportunities
            .checked_add(physical_opportunities)
            .context("owner opportunity overflow")?;
        Ok(
            if total == self.request.owner.budget.maximum_opportunities {
                Terminal::BudgetStop
            } else {
                Terminal::Completed
            },
        )
    }
}
