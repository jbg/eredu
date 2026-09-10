use super::*;

pub(super) fn scaled_rotary(
    input: &NumericTensor,
    rotary: &NumericRotary,
    offset: i32,
) -> Result<NumericTensor, Error> {
    use eredu_nn::RotaryAlgorithm;
    let dims = rotary.dimensions;
    let half = dims as usize / 2;
    let sequence = input.dim(input.shape.len() - 2);
    let mut amplitude = 1.0;
    let mut frequencies = (0..half)
        .map(|i| rotary.base.powf(2.0 * i as f32 / dims as f32).recip())
        .collect::<Vec<_>>();
    match rotary.algorithm {
        RotaryAlgorithm::Default => {}
        RotaryAlgorithm::Linear { factor } => frequencies.iter_mut().for_each(|f| *f /= factor),
        RotaryAlgorithm::Yarn {
            factor,
            original_max_positions,
            beta_fast,
            beta_slow,
            amplitude: scale,
            truncate,
        } => {
            amplitude = scale;
            let correction = |rotations: f32| {
                dims as f32
                    * ((original_max_positions as f32) / (rotations * 2.0 * std::f32::consts::PI))
                        .ln()
                    / (2.0 * rotary.base.ln())
            };
            let low = if truncate {
                correction(beta_fast).floor()
            } else {
                correction(beta_fast)
            }
            .max(0.0);
            let high = if truncate {
                correction(beta_slow).ceil()
            } else {
                correction(beta_slow)
            }
            .min((dims - 1) as f32);
            let width = if low == high { 0.001 } else { high - low };
            for (index, f) in frequencies.iter_mut().enumerate() {
                let ramp = ((index as f32 - low) / width).clamp(0.0, 1.0);
                *f = *f * (1.0 - ramp) + *f / factor * ramp;
            }
        }
        RotaryAlgorithm::Llama3 {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_positions,
        } => {
            let context = original_max_positions as f32;
            for f in &mut frequencies {
                let wavelength = 2.0 * std::f32::consts::PI / *f;
                if wavelength > context / low_frequency_factor {
                    *f /= factor;
                } else if wavelength >= context / high_frequency_factor {
                    let smooth = (context / wavelength - low_frequency_factor)
                        / (high_frequency_factor - low_frequency_factor);
                    *f = (1.0 - smooth) * (*f / factor) + smooth * *f;
                }
            }
        }
        RotaryAlgorithm::Proportional {
            factor,
            rotary_fraction,
        } => {
            let active = ((dims as f32 * rotary_fraction).round() as usize / 2).clamp(1, half);
            for (i, f) in frequencies.iter_mut().enumerate() {
                *f = if i < active { *f * factor } else { 0.0 };
            }
        }
    }
    let mut cosine = NumericTensor::zeros(vec![sequence, half as i32]);
    let mut sine = cosine.clone();
    for position in 0..sequence as usize {
        for (i, f) in frequencies.iter().enumerate() {
            let angle = (offset as f32 + position as f32) * f;
            cosine.data[position * half + i] = angle.cos() * amplitude;
            sine.data[position * half + i] = angle.sin() * amplitude;
        }
    }
    rotary_embeddings(input, dims, rotary.traditional, &cosine, &sine)
}
