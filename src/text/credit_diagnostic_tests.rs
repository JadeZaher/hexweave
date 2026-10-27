use super::credit::{anchor, AnchorRequest};
use super::credit_metrics::{measure, Metrics};
use super::execution::Counts;
use super::history::History;
use super::node::Frame;
use super::shared::Shared;
use super::storage::Store;
use super::{RouteCredit, TILES, VOCAB, WIDTH};
use crate::ternary::{Evidence, TernaryMatrix};
use anyhow::Result;

#[test]
fn cancellation_saturation_and_zero_denominators_are_explicit() -> Result<()> {
    let weights = TernaryMatrix::from_values(1, 4, &[0, -1, 1, 0])?;
    let task = [1.0, 1.0, 0.0, -1.0];
    let route = [-1.0, -2.0, 1.0, 2.0];
    let coupled = [0.0, -1.0, 1.0, 1.0];
    let metrics = measure(&weights, &task, &route, &coupled, &coupled, 2)?;
    assert_eq!(
        (
            metrics.frames,
            metrics.repeated_visit_commits,
            metrics.coordinate_opportunities
        ),
        (2, 1, 4)
    );
    assert_eq!(
        (
            metrics.task_nonzero,
            metrics.route_nonzero,
            metrics.both_nonzero,
            metrics.opposite_sign
        ),
        (3, 4, 3, 3)
    );
    assert_eq!(
        (
            metrics.task_transitions,
            metrics.coupled_transitions,
            metrics.different_results
        ),
        (2, 3, 4)
    );
    assert_eq!(
        (
            metrics.task_opposing,
            metrics.suppressed_task,
            metrics.new_where_task_zero
        ),
        (2, 1, 1)
    );
    assert_eq!(
        (
            metrics.task_descent,
            metrics.coupled_projection,
            metrics.deficit
        ),
        (2.0, 2.0, 4.0)
    );
    assert_eq!(metrics.residual_max, 0.0);
    let mut applied = weights.clone();
    assert_eq!(
        applied.apply_gradient(&mut Evidence::new(4)?, &coupled, 1)?,
        metrics.actual_transitions
    );
    let empty = Metrics::default().report();
    assert!(empty["ratios"]
        .as_object()
        .unwrap()
        .values()
        .all(|value| value.is_null()));
    assert!(measure(&weights, &[f32::NAN; 4], &route, &coupled, &coupled, 1).is_err());
    Ok(())
}

#[test]
fn repeated_anchor_diagnostics_preserve_exact_state_in_both_arms() -> Result<()> {
    for arm in [RouteCredit::Coupled, RouteCredit::RouteToUDetached] {
        let (off, off_shared, off_counts) = repeated_anchor(arm, false)?;
        let (on, on_shared, on_counts) = repeated_anchor(arm, true)?;
        assert_eq!(off.versions, on.versions);
        assert_eq!(off_shared.snapshot(), on_shared.snapshot());
        assert_eq!(
            serde_json::to_value(&off_counts)?,
            serde_json::to_value(&on_counts)?
        );
        assert_eq!(
            serde_json::to_value(&off.io)?,
            serde_json::to_value(&on.io)?
        );
        for id in 0..TILES {
            let file = format!("tile-{id}-v{}.json", off.versions[id]);
            assert_eq!(
                std::fs::read(off.directory.join(&file))?,
                std::fs::read(on.directory.join(&file))?
            );
        }
        let diagnostic = on_counts.credit.as_ref().unwrap();
        assert_eq!((diagnostic.anchors, diagnostic.total.frames), (1, 3));
        assert_eq!(
            (
                diagnostic.total.unique_tile_commits,
                diagnostic.total.repeated_visit_commits
            ),
            (2, 1)
        );
        assert_eq!(diagnostic.total.coordinate_opportunities, 1024);
        assert_eq!(
            diagnostic.total.applied_transitions,
            on_counts.transitions[3]
        );
    }
    Ok(())
}

fn repeated_anchor(arm: RouteCredit, enabled: bool) -> Result<(Store, Shared, Counts)> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let directory = std::env::current_dir()?
        .join("work")
        .join(format!("t2-repeat-{}-{unique}", std::process::id()));
    let mut store = Store::create(&directory, 7)?;
    let mut shared = Shared::new(7)?;
    let frames = repeated_frames(&mut store, &shared)?;
    let mut counts = Counts {
        credit: enabled.then(Default::default),
        ..Counts::default()
    };
    anchor(
        &mut shared,
        &mut store,
        &mut History::default(),
        &mut counts,
        AnchorRequest {
            context: b"bees fly",
            target: b'e' as usize,
            signal: &[0.0; WIDTH],
            frames: &frames,
            baseline: (VOCAB as f32).ln(),
            step: 0,
            predictor: true,
            route_credit: arm,
        },
    )?;
    Ok((store, shared, counts))
}

fn repeated_frames(store: &mut Store, shared: &Shared) -> Result<Vec<Frame>> {
    let center = store
        .mesh
        .coordinates()
        .iter()
        .position(|point| point.q == 0 && point.r == 0)
        .unwrap();
    let path = [center, store.mesh.neighbors(center)?[0], center];
    let mut hidden = shared.input(b"bees fly")?;
    let mut frames = Vec::new();
    for (index, &id) in path.iter().enumerate() {
        let neighbors = store.mesh.neighbors(id)?;
        let frame = store.with_node(id, |node| {
            let mut frame = node.forward(id, &hidden, &[0.0; WIDTH])?;
            if index < 2 {
                let mut route = node.route(&frame, &[0.0; WIDTH], &neighbors, &mut Some(9))?;
                route.selected = route
                    .destinations
                    .iter()
                    .position(|&next| next == path[index + 1])
                    .unwrap();
                frame.route = Some(route);
            }
            Ok(frame)
        })?;
        hidden.clone_from(&frame.output);
        frames.push(frame);
    }
    Ok(frames)
}
