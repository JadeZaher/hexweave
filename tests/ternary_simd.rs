use hexweave::ternary::{avx2_available, TernaryMatrix};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn vector(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| match index % 8 {
            0 => -0.0,
            1 => 0.1,
            2 => -0.3,
            3 => 16_777_216.0,
            4 => -16_777_216.0,
            5 => f32::MIN_POSITIVE,
            6 => f32::from_bits(1),
            _ => 1.0 / 7.0,
        })
        .collect()
}

#[test]
fn dispatch_and_avx2_preserve_scalar_reduction_bits() -> anyhow::Result<()> {
    for seed in [0, 7, u64::MAX, 1 << 32] {
        for (rows, cols) in [(1, 1), (3, 5), (8, 8), (9, 11), (17, 19), (64, 127)] {
            let matrix = TernaryMatrix::new(rows, cols, seed)?;
            let forward_input = vector(cols);
            let transpose_input = vector(rows);
            let forward = bits(&matrix.forward_scalar(&forward_input)?);
            let transpose = bits(&matrix.transpose_scalar(&transpose_input)?);
            assert_eq!(bits(&matrix.forward(&forward_input)?), forward);
            assert_eq!(bits(&matrix.transpose(&transpose_input)?), transpose);
            if avx2_available() {
                assert_eq!(bits(&matrix.forward_avx2(&forward_input)?), forward);
                assert_eq!(bits(&matrix.transpose_avx2(&transpose_input)?), transpose);
            }
        }
    }
    Ok(())
}

#[test]
fn packed_round_trip_and_columns_cover_partial_bytes() -> anyhow::Result<()> {
    for (rows, cols) in [(1, 1), (2, 3), (7, 11), (8, 16)] {
        let matrix = TernaryMatrix::new(rows, cols, u64::MAX)?;
        assert_eq!(
            matrix,
            TernaryMatrix::from_packed(rows, cols, matrix.packed_bytes())?
        );
        for col in 0..cols {
            let column = matrix.column(col)?;
            for (row, value) in column.iter().enumerate() {
                assert_eq!(*value, f32::from(matrix.value(row * cols + col)?));
            }
        }
        assert!(matrix.column(cols).is_err());
    }
    for (rows, cols, bytes) in [
        (1, 1, vec![]),
        (1, 1, vec![0, 0]),
        (1, 1, vec![3]),
        (1, 1, vec![4]),
        (1, 3, vec![64]),
        (1, 4, vec![192]),
        (0, 1, vec![]),
        (usize::MAX, 2, vec![0]),
    ] {
        assert!(TernaryMatrix::from_packed(rows, cols, &bytes).is_err());
    }
    Ok(())
}

#[test]
fn kernels_reject_nonfinite_input_and_overflow() -> anyhow::Result<()> {
    let matrix = TernaryMatrix::from_values(9, 9, &[1; 81])?;
    assert!(matrix.forward_scalar(&[f32::NAN; 9]).is_err());
    assert!(matrix.transpose_scalar(&[f32::INFINITY; 9]).is_err());
    assert!(matrix.forward_scalar(&[f32::MAX; 9]).is_err());
    assert!(matrix.transpose_scalar(&[f32::MAX; 9]).is_err());
    assert!(matrix.forward(&[0.0; 8]).is_err());
    assert!(matrix.transpose(&[0.0; 8]).is_err());
    assert!(matrix.forward_avx2(&[f32::NAN; 9]).is_err());
    assert!(matrix.transpose_avx2(&[f32::INFINITY; 9]).is_err());
    assert!(matrix.forward_avx2(&[f32::MAX; 9]).is_err());
    assert!(matrix.transpose_avx2(&[f32::MAX; 9]).is_err());
    if !avx2_available() {
        assert!(matrix.forward_avx2(&[0.0; 9]).is_err());
        assert!(matrix.transpose_avx2(&[0.0; 9]).is_err());
    }
    Ok(())
}
