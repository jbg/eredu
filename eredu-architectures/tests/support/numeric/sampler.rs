#[derive(Debug, Clone, Eq, PartialEq)]
struct StatefulNumericSampler {
    calls: usize,
    invalid: bool,
}

impl Sampler<NumericBackend> for StatefulNumericSampler {
    fn sample(
        &mut self,
        logits: &NumericTensor,
        temperature: f32,
        random: Option<&mut i32>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.calls += 1;
        if self.invalid {
            if let Some(random) = random {
                *random += 1;
            }
            return Ok(NumericTensor::token_ids(&[usize::MAX / 2]));
        }
        NumericBackend::sample_raw(logits, temperature, random, context)
    }
}
