use super::execution::evaluate;
use super::history::{History, Record};
use super::node::{Gradients, Node, Route};
use super::report::Counts;
use super::state::Model;
use super::{FEATURES, TILES, WIDTH};
use anyhow::Result;

#[test]
fn returned_input_adjoint_matches_finite_differences() -> Result<()> {
    let node = Node::new(42, 4)?;
    let input = vec![-0.3, 0.2, 0.4, -0.1, 0.5, 0.2, -0.4, 0.1];
    let correction = vec![0.2, -0.4, 0.1, 0.3, -0.1, 0.2, 0.3, -0.2];
    let frame = node.forward(0, &input)?;
    let analytic = node.returned_correction(&frame, &correction)?;
    let objective = |values: &[f32]| -> Result<f32> {
        Ok(node
            .forward(0, values)?
            .output
            .iter()
            .zip(&correction)
            .map(|(a, b)| a * b)
            .sum())
    };
    for index in 0..WIDTH {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[index] += 0.001;
        minus[index] -= 0.001;
        let numerical = (objective(&plus)? - objective(&minus)?) / 0.002;
        assert!((analytic[index] - numerical).abs() < 0.0002);
    }
    Ok(())
}

#[test]
fn transform_and_activation_relaxation_gradients_match_finite_differences() -> Result<()> {
    let node = Node::new(42, 4)?;
    let frame = node.forward(0, &[0.3; WIDTH])?;
    let correction = vec![0.4; WIDTH];
    let mut gradients = Gradients::zeros(node.router.len());
    gradients.add_transform(&frame, &correction);
    for (row, &output_correction) in correction.iter().enumerate() {
        for column in 0..WIDTH {
            let delta = 0.001 * frame.input[column] / (WIDTH as f32).sqrt();
            let difference = (frame.preactivation[row] + delta).tanh()
                - (frame.preactivation[row] - delta).tanh();
            let numerical = output_correction * 0.25 * frame.coefficients[row] * difference / 0.002;
            assert!((gradients.transform[row * WIDTH + column] - numerical).abs() < 0.0001);
        }
        let plus = (frame.coefficients[row] + 0.25 * 0.001) * frame.preactivation[row].tanh();
        let minus = (frame.coefficients[row] - 0.25 * 0.001) * frame.preactivation[row].tanh();
        let numerical = output_correction * 0.25 * (plus - minus) / 0.002;
        assert!((gradients.activation[row] - numerical).abs() < 0.0001);
    }
    Ok(())
}

#[test]
fn worse_than_baseline_route_gets_a_downward_selected_logit_update() {
    let route = Route {
        rows: vec![0, TILES],
        probabilities: vec![0.25, 0.75],
        selected: 0,
        features: vec![1.0; 2 * WIDTH],
    };
    let mut gradients = Gradients::zeros(2 * 2 * WIDTH);
    gradients.add_route(&route, 2.0);
    assert!(gradients.router[0] > 0.0);
    assert!(gradients.router[2 * WIDTH] < 0.0);
}

fn record(model: &Model) -> Record {
    Record {
        node: model.center(),
        node_version: 0,
        shared_version: 0,
        step: 0,
        dependency_versions: [0; TILES],
        features: vec![0.0; FEATURES],
        correction: vec![0.2; WIDTH],
    }
}

#[test]
fn any_path_dependency_can_expire_a_returned_correction() -> Result<()> {
    let mut model = Model::new(7)?;
    let mut history = History::new();
    history.push(record(&model));
    model.nodes[0].version = 1;
    history.prune(&model, 1);
    assert!(history.latest().is_some());
    model.nodes[0].version = 2;
    history.prune(&model, 2);
    assert!(history.latest().is_none());
    assert_eq!(history.incompatible_drops, 1);
    Ok(())
}

#[test]
fn correction_history_has_capacity_and_age_bounds() -> Result<()> {
    let model = Model::new(7)?;
    let mut history = History::new();
    for _ in 0..10 {
        history.push(record(&model));
    }
    assert_eq!(history.peak, 8);
    assert_eq!(history.capacity_drops, 2);
    history.prune(&model, 9);
    assert!(history.latest().is_none());
    assert_eq!(history.incompatible_drops, 8);
    Ok(())
}

#[test]
fn task_targets_do_not_change_the_target_free_predictor_or_route() -> Result<()> {
    let model = Model::new(7)?;
    let id = model.center();
    let frame = model.nodes[id].forward(id, &model.input(2)?)?;
    let context = [0.0; WIDTH];
    let features = model.features(&frame, &context);
    let prediction = model.predict(&features)?;
    let neighbors = model.mesh.neighbors(id)?;
    let route = model.nodes[id].route(&frame.output, &context, &neighbors, None)?;
    model.task(&frame.output, 3)?;
    model.task(&frame.output, 9)?;
    assert_eq!(prediction, model.predict(&features)?);
    assert_eq!(
        route.selected,
        model.nodes[id]
            .route(&frame.output, &context, &neighbors, None)?
            .selected
    );
    assert!(model.nodes[id].forward(id, &[f32::NAN; WIDTH]).is_err());
    assert!(model.predict(&[f32::INFINITY; FEATURES]).is_err());
    Ok(())
}

#[test]
fn heldout_evaluation_is_repeatable_and_cannot_mutate_weights() -> Result<()> {
    let model = Model::new(7)?;
    let snapshot = || -> Result<Vec<i8>> {
        model
            .all_matrices()
            .iter()
            .flat_map(|matrix| (0..matrix.len()).map(move |index| matrix.value(index)))
            .collect()
    };
    let before = snapshot()?;
    let first = evaluate(&model, &mut Counts::default())?;
    let second = evaluate(&model, &mut Counts::default())?;
    assert_eq!(first, second);
    assert_eq!(before, snapshot()?);
    assert_eq!(model.shared_version, 0);
    assert!(model.nodes.iter().all(|node| node.version == 0));
    Ok(())
}
