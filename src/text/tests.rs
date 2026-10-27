use super::execution::{execute, Counts, ForwardRequest};
use super::gradients::Gradients;
use super::history::History;
use super::node::{Frame, Node};
use super::shared::Shared;
use super::storage::Store;
use super::{RouteCredit, TILES, WIDTH};
use anyhow::Result;

#[test]
fn joint_tile_task_derivatives_match_finite_differences() -> Result<()> {
    let node = Node::new(7, 4)?;
    let input: Vec<_> = (0..WIDTH).map(|i| 0.1 + i as f32 * 0.031).collect();
    let correction: Vec<_> = (0..WIDTH).map(|i| (i as f32 - 7.0) * 0.02).collect();
    let signal = [0.03; WIDTH];
    let frame = node.forward(0, &input, &signal)?;
    let returned = node.returned(&frame, &correction)?;
    let mut gradients = Gradients::zeros(&node);
    gradients.add(&node, &frame, &correction, None, RouteCredit::Coupled)?;
    let epsilon = 0.001;
    for index in 0..WIDTH {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let numeric = (continuous_loss(&node, &plus, &correction, None)?
            - continuous_loss(&node, &minus, &correction, None)?)
            / (2.0 * epsilon);
        assert!(
            (numeric - returned[index]).abs() < 0.0001,
            "input derivative {index}: {numeric} vs {}",
            returned[index]
        );
    }
    for (kind, values) in [
        (0, &gradients.transform),
        (1, &gradients.curve),
        (2, &gradients.pulse_context),
        (3, &gradients.pulse_heads),
    ] {
        for index in [0, values.len() / 2, values.len() - 1] {
            let numeric =
                (continuous_loss(&node, &input, &correction, Some((kind, index, epsilon)))?
                    - continuous_loss(&node, &input, &correction, Some((kind, index, -epsilon)))?)
                    / (2.0 * epsilon);
            assert!(
                (numeric - values[index]).abs() < 0.0001,
                "parameter derivative {kind}/{index}: {numeric} vs {}",
                values[index]
            );
        }
    }
    Ok(())
}

fn continuous_loss(
    node: &Node,
    input: &[f32],
    correction: &[f32],
    perturb: Option<(usize, usize, f32)>,
) -> Result<f32> {
    let heads = continuous_heads(node, input, perturb)?;
    let mut loss = 0.0;
    for row in 0..WIDTH {
        let mut transformed = 0.0;
        for (col, value) in input.iter().enumerate() {
            let index = row * WIDTH + col;
            transformed +=
                weight(perturb, 0, index, node.transform.value(index)? as f32) * value / 4.0;
        }
        let z = transformed * (1.0 + 0.125 * heads[row].tanh()) + 0.03 / 8.0;
        let (segment, fraction) = if z < 0.0 {
            (0, (z + 2.0) / 2.0)
        } else {
            (1, z / 2.0)
        };
        let left = segment * WIDTH + row;
        let right = (segment + 1) * WIDTH + row;
        let left_value =
            segment as f32 - 1.0 + weight(perturb, 1, left, node.curve.value(left)? as f32) / 4.0;
        let right_value =
            segment as f32 + weight(perturb, 1, right, node.curve.value(right)? as f32) / 4.0;
        loss += correction[row]
            * (input[row]
                + (left_value * (1.0 - fraction)
                    + right_value * fraction
                    + 0.125 * heads[WIDTH + row].tanh())
                    / 4.0);
    }
    Ok(loss)
}

fn weight(perturb: Option<(usize, usize, f32)>, kind: usize, index: usize, base: f32) -> f32 {
    base + perturb
        .filter(|(k, i, _)| *k == kind && *i == index)
        .map_or(0.0, |(_, _, delta)| delta)
}

fn continuous_heads(
    node: &Node,
    input: &[f32],
    perturb: Option<(usize, usize, f32)>,
) -> Result<Vec<f32>> {
    let features: Vec<_> = input.iter().copied().chain([0.03; WIDTH]).collect();
    let mut latent = [0.0; WIDTH];
    for (row, output) in latent.iter_mut().enumerate() {
        for (col, value) in features.iter().enumerate() {
            let index = row * 2 * WIDTH + col;
            *output += weight(perturb, 2, index, node.pulse.context.value(index)? as f32) * value;
        }
        *output = (*output / (2.0 * WIDTH as f32).sqrt()).tanh();
    }
    let mut heads = vec![0.0; node.pulse.heads.rows()];
    for (row, output) in heads.iter_mut().enumerate() {
        for (col, value) in latent.iter().enumerate() {
            let index = row * WIDTH + col;
            *output +=
                weight(perturb, 3, index, node.pulse.heads.value(index)? as f32) * value / 4.0;
        }
    }
    Ok(heads)
}

#[test]
fn routing_credit_reaches_shared_trunk_and_changes_other_heads() -> Result<()> {
    let mut node = Node::new(7, 4)?;
    let input: Vec<_> = (0..WIDTH).map(|i| 0.1 + i as f32 * 0.031).collect();
    let mut frame = node.forward(0, &input, &[0.03; WIDTH])?;
    frame.route = Some(node.route(&frame, &[0.03; WIDTH], &[1, 2, 3], &mut Some(9))?);
    let mut task_only = Gradients::zeros(&node);
    task_only.add(&node, &frame, &[0.1; WIDTH], None, RouteCredit::Coupled)?;
    assert!(task_only
        .pulse_context
        .iter()
        .any(|value| value.abs() > 0.00001));
    assert!(task_only.pulse_heads[2 * WIDTH * WIDTH..]
        .iter()
        .all(|value| *value == 0.0));
    let mut gradients = Gradients::zeros(&node);
    gradients.add(
        &node,
        &frame,
        &[0.0; WIDTH],
        Some(0.7),
        RouteCredit::Coupled,
    )?;
    assert!(gradients
        .pulse_context
        .iter()
        .any(|value| value.abs() > 0.00001));
    assert!(gradients.pulse_heads[..2 * WIDTH * WIDTH]
        .iter()
        .all(|value| *value == 0.0));
    check_detached_credit(&node, &frame, &task_only, &gradients)?;
    let epsilon = 0.005;
    for (kind, values) in [(2, &gradients.pulse_context), (3, &gradients.pulse_heads)] {
        for index in [0, values.len() / 2, values.len() - 1] {
            let numeric = (continuous_policy(&node, &frame, Some((kind, index, epsilon)))?
                - continuous_policy(&node, &frame, Some((kind, index, -epsilon)))?)
                / (2.0 * epsilon);
            assert!(
                (numeric - values[index]).abs() < 0.0001,
                "policy derivative {kind}/{index}: {numeric} vs {}",
                values[index]
            );
        }
    }
    let transitions = node.apply(&gradients)?;
    assert!(transitions[3] > 0 && transitions[4] > 0);
    let after = node.forward(0, &input, &[0.03; WIDTH])?;
    assert_ne!(&frame.pulse.heads[..WIDTH], &after.pulse.heads[..WIDTH]);
    assert_ne!(
        &frame.pulse.heads[WIDTH..2 * WIDTH],
        &after.pulse.heads[WIDTH..2 * WIDTH]
    );
    Ok(())
}

fn check_detached_credit(
    node: &Node,
    frame: &Frame,
    task: &Gradients,
    route: &Gradients,
) -> Result<()> {
    let mut detached = Gradients::zeros(node);
    detached.add(
        node,
        frame,
        &[0.0; WIDTH],
        Some(0.7),
        RouteCredit::RouteToUDetached,
    )?;
    assert!(detached.pulse_context.iter().all(|value| *value == 0.0));
    assert_eq!(detached.pulse_heads, route.pulse_heads);
    assert_eq!(detached.router, route.router);
    assert!(detached.router.iter().any(|value| *value != 0.0));
    let mut mixed = Gradients::zeros(node);
    mixed.add(
        node,
        frame,
        &[0.1; WIDTH],
        Some(0.7),
        RouteCredit::RouteToUDetached,
    )?;
    assert_eq!(mixed.pulse_context, task.pulse_context);
    assert_eq!(mixed.transform, task.transform);
    assert_eq!(mixed.curve, task.curve);
    assert_eq!(mixed.router, route.router);
    assert_eq!(
        &mixed.pulse_heads[..2 * WIDTH * WIDTH],
        &task.pulse_heads[..2 * WIDTH * WIDTH]
    );
    assert_eq!(
        &mixed.pulse_heads[2 * WIDTH * WIDTH..],
        &route.pulse_heads[2 * WIDTH * WIDTH..]
    );
    Ok(())
}

fn continuous_policy(
    node: &Node,
    frame: &Frame,
    perturb: Option<(usize, usize, f32)>,
) -> Result<f32> {
    let heads = continuous_heads(node, &frame.input, perturb)?;
    let route = frame.route.as_ref().unwrap();
    let mut logits = vec![0.0; node.router.rows()];
    for (row, logit) in logits.iter_mut().enumerate() {
        for (col, value) in route.features.iter().enumerate() {
            *logit += node.router.value(row * 2 * WIDTH + col)? as f32 * value
                / (2.0 * WIDTH as f32).sqrt();
        }
        *logit += 0.25 * heads[2 * WIDTH + row];
    }
    Ok(0.7 * super::math::softmax(&logits)?[route.selected].ln())
}

#[test]
fn local_updates_leave_no_tape_and_only_touch_visited_tiles() -> Result<()> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let directory = std::env::current_dir()?
        .join("work")
        .join(format!("text-local-test-{}-{unique}", std::process::id()));
    let mut store = Store::create(&directory, 7)?;
    let shared = Shared::new(7)?;
    let mut counts = Counts::default();
    let frames = execute(
        &shared,
        &mut store,
        &mut History::default(),
        &mut counts,
        ForwardRequest {
            context: b"bees fly",
            history: &[0.0; WIDTH],
            random_seed: Some(1),
            local: true,
            step: 1,
        },
    )?;
    assert!(frames.is_empty());
    assert!((1..=3).contains(&counts.tile_visits));
    for id in 0..TILES {
        if counts.visits_by_tile[id] == 0 {
            assert_eq!(store.versions[id], 0);
        }
    }
    assert_eq!(store.io.peak_live_tiles, 1);
    Ok(())
}
