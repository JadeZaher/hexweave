use anyhow::Result;
use candle_core::{Device, Tensor};
use hexweave::{
    config::Config,
    model::{cross_entropy, HexModel},
    smoke::run_smoke,
};

#[test]
fn future_tokens_do_not_change_prefix_logits_or_route() -> Result<()> {
    let model = HexModel::new(Config::smoke(), 42)?;
    let history = model.empty_history()?;
    let left = model.forward(&[1, 2, 3, 4, 5, 6, 7, 8], &history, Some(99))?;
    let right = model.forward(&[1, 2, 3, 4, 19, 18, 17, 16], &history, Some(99))?;
    assert_eq!(left.visited, right.visited);
    let difference = left
        .logits
        .narrow(0, 0, 4)?
        .sub(&right.logits.narrow(0, 0, 4)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(difference < 2e-5, "prefix changed by {difference}");
    Ok(())
}

#[test]
fn realized_path_backpropagates_into_embedding_curve_and_pulse() -> Result<()> {
    let model = HexModel::new(Config::smoke(), 42)?;
    let pass = model.forward(&[1, 2, 3, 4], &model.empty_history()?, Some(17))?;
    let gradients = cross_entropy(&pass.logits, &[2, 3, 4, 5])?.backward()?;
    for family in ["embedding", "pulse_input", "curve_knots"] {
        let magnitude: f32 = model
            .parameters()
            .iter()
            .filter(|(name, _)| name.contains(family))
            .filter_map(|(_, variable)| gradients.get(variable.as_tensor()))
            .map(|gradient| gradient.abs()?.sum_all()?.to_scalar::<f32>())
            .collect::<candle_core::Result<Vec<_>>>()?
            .iter()
            .sum();
        assert!(
            magnitude.is_finite() && magnitude > 0.0,
            "no {family} gradient"
        );
    }
    Ok(())
}

#[test]
fn training_updates_router_curve_and_pulse_with_finite_metrics() -> Result<()> {
    let report = run_smoke(8, 42)?;
    for family in ["router", "curve", "pulse", "other"] {
        let changed = report["parameter_l1_change"][family].as_f64().unwrap();
        assert!(
            changed.is_finite() && changed > 0.0,
            "{family} did not update"
        );
    }
    assert!(report["initial_loss"].as_f64().unwrap().is_finite());
    assert!(report["final_loss"].as_f64().unwrap().is_finite());
    assert!(run_smoke(0, 42).is_err());
    assert!(run_smoke(17, 42).is_err());
    Ok(())
}

#[test]
fn numerical_backend_rejects_large_or_fanout_allocations() -> Result<()> {
    let mut config = Config::smoke();
    config.hidden = 768;
    assert!(HexModel::new(config, 42).is_err());
    let mut config = Config::smoke();
    config.fanout = true;
    assert!(HexModel::new(config, 42).is_err());
    Ok(())
}

#[test]
fn invalid_history_and_nonfinite_router_are_rejected() -> Result<()> {
    let model = HexModel::new(Config::smoke(), 42)?;
    let history = Tensor::from_vec(vec![f32::NAN; 8], (1, 8), &Device::Cpu)?;
    assert!(model.forward(&[1, 2], &history, Some(7)).is_err());
    let history = Tensor::from_vec(vec![0_f64; 8], (1, 8), &Device::Cpu)?;
    assert!(model.forward(&[1, 2], &history, Some(7)).is_err());
    let router = model
        .parameters()
        .into_iter()
        .find(|(name, _)| name == "router")
        .unwrap()
        .1;
    router.set(&Tensor::from_vec(
        vec![f32::NAN; router.elem_count()],
        router.shape(),
        &Device::Cpu,
    )?)?;
    assert!(model
        .forward(&[1, 2], &model.empty_history()?, Some(7))
        .is_err());
    Ok(())
}

#[test]
fn fixed_nonzero_history_preserves_prefix_causality_at_adversarial_seed() -> Result<()> {
    let model = HexModel::new(Config::smoke(), 0x9e37_79b9_7f4a_7c15)?;
    let history = Tensor::from_vec(vec![0.25_f32; 8], (1, 8), &Device::Cpu)?;
    let left = model.forward(&[1, 2, 3, 4], &history, Some(u64::MAX))?;
    let right = model.forward(&[1, 2, 17, 18], &history, Some(u64::MAX))?;
    assert_eq!(left.visited, right.visited);
    let difference = left
        .logits
        .narrow(0, 0, 2)?
        .sub(&right.logits.narrow(0, 0, 2)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(difference < 2e-5);
    Ok(())
}
