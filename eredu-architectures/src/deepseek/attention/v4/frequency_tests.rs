use super::*;
use crate::deepseek::config::YarnConfig;
use eredu_nn::workspace::*;

// The prior allocation-based formula is retained only as an independent
// numerical reference. These vectors are not production constructor payloads.
fn original(dimensions: i32, base: f32, yarn: Option<&YarnConfig>, scale: i32) -> Vec<f32> {
    let mut inverse = (0..dimensions)
        .step_by(2)
        .map(|index| 1.0 / base.powf(index as f32 / dimensions as f32))
        .collect::<Vec<_>>();
    if let Some(config) = yarn {
        let correction = |rotations: f32| {
            dimensions as f32
                * (config.original_max_position_embeddings as f32
                    / (rotations * 2.0 * std::f32::consts::PI))
                    .ln()
                / (2.0 * base.ln())
        };
        let low = correction(config.beta_fast).floor().max(0.0);
        let mut high = correction(config.beta_slow)
            .ceil()
            .min((dimensions - 1) as f32);
        if low == high {
            high += 0.001;
        }
        for (index, frequency) in inverse.iter_mut().enumerate() {
            let ramp = ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *frequency = *frequency / config.factor * (1.0 - smooth) + *frequency * smooth;
        }
    }
    inverse
        .into_iter()
        .map(|frequency| 1.0 / frequency / scale as f32)
        .collect()
}

#[test]
fn fixed_scalar_equation_preserves_v4_f32_bits_and_yarn_boundaries() {
    for dimensions in [2, 8, 64, 128] {
        for base in [10_000.0, 500_000.0] {
            for scale in [1, 4, 128] {
                for (factor, original_positions, fast, slow) in [
                    (1.0, 128, 32.0, 1.0),
                    (8.0, 4096, 32.0, 1.0),
                    (16.0, 1, 32.0, 32.0),
                ] {
                    let config = YarnConfig {
                        r#type: "yarn".into(),
                        factor,
                        original_max_position_embeddings: original_positions,
                        beta_fast: fast,
                        beta_slow: slow,
                        mscale: 1.0,
                        mscale_all_dim: 0.0,
                    };
                    for yarn in [None, Some(&config)] {
                        let program = V4FrequencyValues::new(dimensions, base, yarn, scale);
                        let expected = original(dimensions, base, yarn, scale);
                        let actual = (0..dimensions as usize / 2)
                            .map(|i| program.value(i).to_bits())
                            .collect::<Vec<_>>();
                        assert_eq!(
                            actual,
                            expected.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                            "dimensions={dimensions}, base={base}, scale={scale}, yarn={yarn:?}"
                        );
                        assert!(expected.iter().all(|f| f.is_finite() && *f > 0.0));
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct ConstructorMechanism;
impl WorkspaceMechanisms for ConstructorMechanism {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        assert!(matches!(
            operation.kind,
            WorkspaceOperationKind::GeneratedF32Initialization
        ));
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![WorkspaceOutputStorage::Allocate(
                operation.outputs[0].bytes()?,
            )],
            scratch_bytes: 0,
            assumptions: "test mechanism copies the generated F32 buffer once".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: eredu_nn::F32InitializationPlan::new(operation.outputs[0].shape())
                .unwrap()
                .host_buffer_bytes(),
            assumptions: "actual shared fixed initializer's one host buffer".into(),
        }))
    }
}

#[test]
fn v4_constructor_uses_shared_fixed_operation_for_each_separate_helper() {
    let context = WorkspaceContext::new(ConstructorMechanism);
    let first: WorkspaceTensor = V4FrequencyValues::new(8, 10_000.0, None, 1)
        .initialize(&context)
        .unwrap();
    let alias = first.clone();
    let second: WorkspaceTensor = V4FrequencyValues::new(8, 10_000.0, None, 4)
        .initialize(&context)
        .unwrap();
    let report = context.report(&[first, alias, second]).unwrap();
    assert_eq!(report.operations.len(), 2);
    assert_eq!(report.host_workspace_bytes, Some(32));
    assert_eq!(report.retained_bytes, Some(32));
    assert_eq!(report.total_bytes, Some(64));
}
