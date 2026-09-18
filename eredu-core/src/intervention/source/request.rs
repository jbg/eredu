//! Fresh request intermediates use the same typed copy as admitted sources.
use super::*;

/// Borrowed description of the original declaration copy producer. Inspection
/// allocates no output; the caller pays its control layout before inspection and
/// the complete returned extent before invoking the construction worker.
///
/// The copied DTO is a construction intermediate, never an admitted source. Its
/// enclosing admission or record producer must retain the supplied destination
/// through payload, partial failure and result retirement. No existing buffer is
/// adopted and this program issues no execution or source identity authority.
#[derive(Debug)]
pub struct PreparedInterventionRequestCopy<'a> {
    source: &'a InterventionPlan,
    bytes: usize,
}
impl<'a> PreparedInterventionRequestCopy<'a> {
    fn controls() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Worker>(),
            size_of::<InterventionPlan>() * 2,
            size_of::<InterventionOperation>(),
            size_of::<InterventionAction>(),
            size_of::<InterventionTensor>(),
            size_of::<InterventionValues>(),
            size_of::<CaptureSlice>(),
            size_of::<CaptureSchedule>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<CapturePlanCopyError>(),
            size_of::<Result<InterventionPlan, CapturePlanCopyError>>(),
            size_of::<Result<Self, CapturePlanCopyError>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Exact fixed nonallocating query frames, including the typed nested copy.
    pub fn inspection_control_bytes() -> Option<usize> {
        Self::controls()
    }
    /// Measures the same exhaustive request worker without emitting destinations.
    pub fn inspect(source: &'a InterventionPlan) -> Result<Self, CapturePlanCopyError> {
        let mut worker = Worker::with_controls(
            false,
            Self::controls().ok_or(CapturePlanCopyError::Overflow)?,
        );
        drop(copy::request(&mut worker, source)?);
        Ok(Self {
            source,
            bytes: worker.bytes(),
        })
    }
    /// Actual selected vector/string destinations and complete fixed controls.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Runs only after the enclosing producer has paid `required_bytes`. The
    /// authority keeps that original destination alive during construction; the
    /// caller must retain it with every escaping DTO or enclosing typed error.
    pub fn copy(
        self,
        _destination: &HostPreparationAuthority,
    ) -> Result<InterventionPlan, CapturePlanCopyError> {
        let mut worker = Worker::with_controls(
            true,
            Self::controls().ok_or(CapturePlanCopyError::Overflow)?,
        );
        let output = copy::request(&mut worker, self.source)?;
        if worker.bytes() != self.bytes {
            return Err(CapturePlanCopyError::Capacity);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_copy_preserves_every_action_and_owns_independent_destinations() {
        use InterventionAction::*;
        let dtype = InterventionDtype::Float32;
        let mut actions = vec![
            Zero { dtype },
            Scale {
                dtype,
                factor: -0.75,
            },
            Mask {
                dtype,
                shape: vec![1, 3],
                keep: vec![true, false, true],
            },
            MaskComponents {
                dtype,
                indices: vec![2, 7],
                keep_selected: true,
            },
            MaskLogits {
                dtype,
                token_ids: vec![0, 17, 99],
            },
            ExcludeExperts {
                expert_ids: vec![1, 5],
            },
            ZeroExpertContribution {
                expert_ids: vec![2],
            },
            BiasRoutingScores {
                stage: RoutingScoreStage::RankingScores,
                expert_ids: vec![1, 3],
                biases: vec![-1.5, 0.25],
            },
            ForceExperts {
                shape: [1, 2],
                expert_ids: vec![3, 7],
            },
        ];
        for values in [
            InterventionValues::Float32(vec![-2.0, 0.125]),
            InterventionValues::Float16(vec![0x3c00, 0xbc00]),
            InterventionValues::Bfloat16(vec![0x3f80, 0xbf80]),
        ] {
            let tensor = InterventionTensor {
                shape: vec![1, 2],
                values,
            };
            actions.push(Replace {
                tensor: tensor.clone(),
            });
            actions.push(Add { tensor });
        }
        let mut source = InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations: actions
                .into_iter()
                .enumerate()
                .map(|(i, action)| InterventionOperation {
                    id: format!("operation-{i}-λ"),
                    target: "block.output".into(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![CaptureSlice {
                        axis: "sequence".into(),
                        start: 1,
                        end: 3,
                        stride: 1,
                    }],
                    action,
                    evidence: InterventionEvidence::Preview { max_elements: 2 },
                })
                .collect(),
        };
        source.operations.reserve(17);
        let prepared = PreparedInterventionRequestCopy::inspect(&source).unwrap();
        assert!(prepared.required_bytes() > std::mem::size_of_val(&source));
        let copied = prepared
            .copy(&HostPreparationAuthority::unmanaged())
            .unwrap();
        assert_eq!(copied, source);
        assert_eq!(copied.operations.capacity(), copied.operations.len());
        for (old, new) in source.operations.iter().zip(&copied.operations) {
            assert_ne!(old.id.as_ptr(), new.id.as_ptr());
            assert_ne!(old.slices.as_ptr(), new.slices.as_ptr());
            assert_ne!(old.slices[0].axis.as_ptr(), new.slices[0].axis.as_ptr());
        }
        let expected = serde_json::to_vec(&source).unwrap();
        source.operations.clear();
        assert_eq!(serde_json::to_vec(&copied).unwrap(), expected);
    }
}
