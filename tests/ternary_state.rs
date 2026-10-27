use hexweave::ternary::{Evidence, TernaryMatrix};

#[test]
fn values_are_packed_and_shape_checked() -> anyhow::Result<()> {
    let matrix = TernaryMatrix::from_values(1, 5, &[-1, 0, 1, -1, 1])?;
    assert_eq!((matrix.rows(), matrix.cols(), matrix.len()), (1, 5, 5));
    assert!(!matrix.is_empty());
    assert_eq!(matrix.storage_bytes(), 2);
    assert_eq!(Evidence::new(5)?.bytes(), 10);
    for (index, expected) in [-1, 0, 1, -1, 1].iter().enumerate() {
        assert_eq!(matrix.value(index)?, *expected);
    }
    assert!(matrix.value(5).is_err());
    assert!(TernaryMatrix::new(0, 1, 0).is_err());
    assert!(TernaryMatrix::new(usize::MAX, 2, 0).is_err());
    assert!(TernaryMatrix::new(1, 1_048_577, 0).is_err());
    assert!(TernaryMatrix::from_values(1, 1, &[2]).is_err());
    assert!(TernaryMatrix::from_values(1, 2, &[0]).is_err());
    assert!(Evidence::new(0).is_err());
    assert!(Evidence::new(1_048_577).is_err());
    Ok(())
}

#[test]
fn deterministic_initialization_uses_the_complete_seed() -> anyhow::Result<()> {
    let low = TernaryMatrix::new(8, 8, 7)?;
    assert_eq!(low, TernaryMatrix::new(8, 8, 7)?);
    assert_ne!(low, TernaryMatrix::new(8, 8, 7 + (1 << 32))?);
    assert_eq!(TernaryMatrix::new(1, 1, u64::MAX)?.len(), 1);
    for index in 0..low.len() {
        assert!((-1..=1).contains(&low.value(index)?));
    }
    Ok(())
}

#[test]
fn forward_and_transpose_are_exact_scalar_products() -> anyhow::Result<()> {
    let matrix = TernaryMatrix::from_values(2, 3, &[1, 0, -1, -1, 1, 1])?;
    assert_eq!(matrix.forward(&[2.0, 3.0, 5.0])?, vec![-3.0, 6.0]);
    assert_eq!(matrix.transpose(&[2.0, 4.0])?, vec![-2.0, 4.0, 2.0]);
    assert!(matrix.forward(&[0.0]).is_err());
    assert!(matrix.forward(&[0.0, f32::NAN, 0.0]).is_err());
    assert!(matrix.transpose(&[0.0]).is_err());
    assert!(matrix.transpose(&[0.0, f32::INFINITY]).is_err());
    let overflowing = TernaryMatrix::from_values(1, 2, &[1, 1])?;
    assert!(overflowing.forward(&[f32::MAX, f32::MAX]).is_err());
    assert!(overflowing.transpose(&[f32::INFINITY]).is_err());
    Ok(())
}

#[test]
fn invalid_update_is_atomic_for_weights_and_evidence() -> anyhow::Result<()> {
    let mut matrix = TernaryMatrix::from_values(1, 2, &[0, 0])?;
    let mut evidence = Evidence::new(2)?;
    matrix.apply_gradient(&mut evidence, &[-1.0, 1.0], 2)?;
    let saved_matrix = matrix.clone();
    let saved_evidence = evidence.clone();
    for gradient in [vec![-1.0], vec![-1.0, f32::NAN], vec![f32::INFINITY, 1.0]] {
        assert!(matrix.apply_gradient(&mut evidence, &gradient, 2).is_err());
        assert_eq!(matrix, saved_matrix);
        assert_eq!(evidence, saved_evidence);
    }
    for threshold in [0, -1, i16::MIN] {
        assert!(matrix
            .apply_gradient(&mut evidence, &[-1.0, 1.0], threshold)
            .is_err());
        assert_eq!(matrix, saved_matrix);
        assert_eq!(evidence, saved_evidence);
    }
    let mut wrong_evidence = Evidence::new(1)?;
    let saved_wrong = wrong_evidence.clone();
    assert!(matrix
        .apply_gradient(&mut wrong_evidence, &[-1.0, 1.0], 2)
        .is_err());
    assert_eq!(matrix, saved_matrix);
    assert_eq!(wrong_evidence, saved_wrong);
    Ok(())
}

#[test]
fn thresholds_move_one_trit_at_a_time_and_reset_at_saturation() -> anyhow::Result<()> {
    let mut matrix = TernaryMatrix::from_values(1, 2, &[-1, 1])?;
    let mut evidence = Evidence::new(2)?;
    assert_eq!(matrix.apply_gradient(&mut evidence, &[-1.0, 1.0], 2)?, 0);
    assert_eq!(matrix.apply_gradient(&mut evidence, &[0.0, -0.0], 2)?, 0);
    assert_eq!(matrix.apply_gradient(&mut evidence, &[-100.0, 0.01], 2)?, 2);
    assert_eq!((matrix.value(0)?, matrix.value(1)?), (0, 0));
    assert_eq!(evidence, Evidence::new(2)?);
    for expected_changes in [0, 2, 0, 0] {
        assert_eq!(
            matrix.apply_gradient(&mut evidence, &[-1.0, 1.0], 2)?,
            expected_changes
        );
    }
    assert_eq!((matrix.value(0)?, matrix.value(1)?), (1, -1));
    assert_eq!(evidence, Evidence::new(2)?);
    assert_eq!(matrix.apply_gradient(&mut evidence, &[1.0, -1.0], 2)?, 0);
    assert_eq!(matrix.apply_gradient(&mut evidence, &[1.0, -1.0], 2)?, 2);
    assert_eq!((matrix.value(0)?, matrix.value(1)?), (0, 0));
    Ok(())
}

#[test]
fn opposite_signs_cancel_and_i16_threshold_does_not_overflow() -> anyhow::Result<()> {
    let mut matrix = TernaryMatrix::from_values(1, 1, &[0])?;
    let mut evidence = Evidence::new(1)?;
    matrix.apply_gradient(&mut evidence, &[1.0], 2)?;
    matrix.apply_gradient(&mut evidence, &[-1.0], 2)?;
    assert_eq!(evidence, Evidence::new(1)?);
    for _ in 1..i16::MAX {
        assert_eq!(matrix.apply_gradient(&mut evidence, &[-1.0], i16::MAX)?, 0);
    }
    assert_eq!(matrix.apply_gradient(&mut evidence, &[-1.0], i16::MAX)?, 1);
    assert_eq!(matrix.value(0)?, 1);
    assert_eq!(evidence, Evidence::new(1)?);
    Ok(())
}
