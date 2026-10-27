//! Adversarial evidence contracts retained after rejecting a slower fast path.

use super::{Evidence, TernaryMatrix};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediate_updates_match_general_rule_with_inherited_evidence() -> anyhow::Result<()> {
        for len in [1, 3, 4, 5, 17, 127] {
            let mut matrix = TernaryMatrix::new(1, len, u64::MAX)?;
            let prior = [i16::MIN, -2, -1, 0, 1, 2, i16::MAX];
            let mut evidence = Evidence {
                values: (0..len).map(|index| prior[index % prior.len()]).collect(),
            };
            let gradients: Vec<_> = (0..len)
                .map(|index| [-f32::MAX, -0.0, 0.0, f32::MAX, 0.1][index % 5])
                .collect();
            let before = matrix.clone();
            let before_evidence = evidence.clone();
            let changed = matrix.apply_gradient(&mut evidence, &gradients, 1)?;
            let mut expected_changes = 0;
            for (index, &gradient) in gradients.iter().enumerate() {
                let direction = if gradient > 0.0 {
                    -1
                } else if gradient < 0.0 {
                    1
                } else {
                    0
                };
                let accumulated = before_evidence.values[index].saturating_add(direction);
                let movement = if accumulated >= 1 {
                    1
                } else if accumulated <= -1 {
                    -1
                } else {
                    0
                };
                let expected = (before.value(index)? + movement).clamp(-1, 1);
                assert_eq!(matrix.value(index)?, expected);
                expected_changes += usize::from(expected != before.value(index)?);
            }
            assert_eq!(changed, expected_changes);
            assert_eq!(evidence, Evidence::new(len)?);
            assert_eq!(
                matrix,
                TernaryMatrix::from_packed(1, len, matrix.packed_bytes())?
            );
        }
        Ok(())
    }

    #[test]
    fn lower_threshold_uses_retained_evidence_even_with_zero_gradient() -> anyhow::Result<()> {
        let mut matrix = TernaryMatrix::from_values(1, 5, &[0, 0, -1, 1, 0])?;
        let mut evidence = Evidence::new(5)?;
        matrix.apply_gradient(&mut evidence, &[-1.0, 1.0, 1.0, -1.0, -1.0], 2)?;
        assert_eq!(
            matrix.apply_gradient(&mut evidence, &[0.0, -0.0, 0.0, 0.0, 1.0], 1)?,
            2
        );
        assert_eq!(
            matrix,
            TernaryMatrix::from_values(1, 5, &[1, -1, -1, 1, 0])?
        );
        assert_eq!(evidence, Evidence::new(5)?);
        let saved_matrix = matrix.clone();
        let saved_evidence = evidence.clone();
        assert!(matrix
            .apply_gradient(&mut evidence, &[0.0, 0.0, 0.0, 0.0, f32::NAN], 1)
            .is_err());
        assert_eq!(matrix, saved_matrix);
        assert_eq!(evidence, saved_evidence);
        Ok(())
    }

    #[test]
    fn stochastic_rate_spans_frozen_to_pinned_sign_step() -> anyhow::Result<()> {
        let len = 4096;
        let gradients: Vec<_> = (0..len)
            .map(|index| [-1.0, 0.0, 1.0, -0.5][index % 4])
            .collect();
        let start = TernaryMatrix::new(1, len, 11)?;
        let mut pinned = start.clone();
        pinned.apply_gradient(&mut Evidence::new(len)?, &gradients, 1)?;
        // 0.25 attempts ~640 of 3072 nonzero entries; a third already sit at the clamp.
        for (rate, low, high) in [(0.0, 0, 0), (1e9, 0, len), (0.25, 340, 520)] {
            let mut matrix = start.clone();
            let moved = matrix.apply_stochastic(&gradients, rate, 5)?;
            assert!((low..=high).contains(&moved), "rate {rate}: {moved} moved");
            if rate == 0.0 {
                assert_eq!(matrix, start);
            }
            if rate > 1.0 {
                assert_eq!(matrix, pinned);
            }
        }
        assert!(start
            .clone()
            .apply_stochastic(&gradients, f64::NAN, 5)
            .is_err());
        Ok(())
    }

    #[test]
    fn latent_steps_integrate_consistent_gradients_and_hold_on_zero() -> anyhow::Result<()> {
        let len = 64;
        let start = TernaryMatrix::new(1, len, 13)?;
        let initial: Vec<f32> = (0..len)
            .map(|i| start.value(i).map(f32::from))
            .collect::<anyhow::Result<_>>()?;
        // Adam's normalized step is ~rate per call, so 0.1 needs >=5 calls to cross a ±0.5 threshold.
        for (gradient, rate, steps, expected) in [
            (1.0, 0.1, 30, Some(-1)),
            (0.0, 0.1, 30, None),
            (1.0, 0.0, 30, None),
        ] {
            let mut matrix = start.clone();
            let (mut latent, mut moments) = (initial.clone(), vec![0.0; 2 * len]);
            let mut first_move = None;
            for step in 1..=steps {
                let moved = matrix.apply_latent(
                    &mut latent,
                    &mut moments,
                    &vec![gradient; len],
                    rate,
                    step,
                )?;
                if moved > 0 && first_move.is_none() {
                    first_move = Some(step);
                }
            }
            match expected {
                Some(trit) => {
                    assert!((0..len).all(|i| matrix.value(i).ok() == Some(trit)));
                    assert!(
                        first_move.is_some_and(|step| step >= 5),
                        "moved at {first_move:?}"
                    );
                }
                None => assert_eq!(matrix, start),
            }
        }
        assert!(start
            .clone()
            .apply_latent(
                &mut initial.clone(),
                &mut vec![0.0; 2 * len],
                &vec![1.0; len],
                0.1,
                0
            )
            .is_err());
        Ok(())
    }
}
