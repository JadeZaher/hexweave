use candle_core::{Device, Tensor, Var};

use super::ops::{curve, log_softmax, rms, Random};

pub(super) struct Tile {
    query: Var,
    key: Var,
    value: Var,
    output: Var,
    gate: Var,
    up: Var,
    down: Var,
    knots: Var,
    curve_condition: Var,
}

impl Tile {
    pub fn new(
        width: usize,
        inner: usize,
        pulse: usize,
        random: &mut Random,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            query: random.matrix(width, width)?,
            key: random.matrix(width, width)?,
            value: random.matrix(width, width)?,
            output: random.matrix(width, width)?,
            gate: random.matrix(width, inner)?,
            up: random.matrix(width, inner)?,
            down: random.matrix(inner, width)?,
            knots: Var::from_vec(vec![-0.6_f32, 0.0, 0.6], (1, 3), &Device::Cpu)?,
            curve_condition: random.matrix(pulse, 3)?,
        })
    }

    pub fn forward(
        &self,
        previous: &Tensor,
        pulse: &Tensor,
        heads: usize,
    ) -> candle_core::Result<Tensor> {
        let attended = previous.add(&self.attention(previous, heads)?.affine(0.2, 0.0)?)?;
        let normalized = rms(&attended)?;
        let gate = normalized.matmul(&self.gate)?;
        let up = normalized.matmul(&self.up)?;
        let knots = pulse
            .matmul(&self.curve_condition)?
            .broadcast_add(&self.knots)?
            .tanh()?
            .affine(2.0, 0.0)?;
        let residual = curve(&gate, &knots)?.mul(&up)?.matmul(&self.down)?;
        attended.add(&residual.affine(0.2, 0.0)?)
    }

    fn attention(&self, previous: &Tensor, heads: usize) -> candle_core::Result<Tensor> {
        let (length, width) = previous.dims2()?;
        let dimension = width / heads;
        let normalized = rms(previous)?;
        let project = |weight: &Var| -> candle_core::Result<Tensor> {
            normalized
                .matmul(weight)?
                .reshape((length, heads, dimension))?
                .transpose(0, 1)?
                .contiguous()
        };
        let query = project(&self.query)?;
        let key = project(&self.key)?;
        let value = project(&self.value)?;
        let scores = query
            .matmul(&key.transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (dimension as f64).sqrt(), 0.0)?;
        let mask: Vec<_> = (0..length)
            .flat_map(|row| {
                (0..length).map(move |column| if column <= row { 0_f32 } else { -1e9_f32 })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (length, length), &Device::Cpu)?;
        let probability = log_softmax(&scores.broadcast_add(&mask)?, 2)?.exp()?;
        probability
            .matmul(&value)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((length, width))?
            .matmul(&self.output)
    }

    pub fn parameters(&self) -> Vec<(&'static str, Var)> {
        vec![
            ("query", self.query.clone()),
            ("key", self.key.clone()),
            ("value", self.value.clone()),
            ("output", self.output.clone()),
            ("gate", self.gate.clone()),
            ("up", self.up.clone()),
            ("down", self.down.clone()),
            ("curve_knots", self.knots.clone()),
            ("curve_condition", self.curve_condition.clone()),
        ]
    }
}
