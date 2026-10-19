//! Retained producer sites for external target captures, resolved without artifacts or devices.
use crate::component_partition::{ComponentPartitionError, ComponentPartitionLayouts};
use eredu_core::{capture::CaptureError, ParallelRankTopology, SymbolicDimension as Dim};
use eredu_runtime::{ForwardCaptureBundle, ForwardCaptureSlot};

/// Exact prepared target placement and ordered complete hidden capture sites.
/// Symbolic batch/sequence dimensions resolve from architecture-admitted input.
#[derive(Debug, Clone)]
pub struct PreparedTargetCapturePlacement {
    topology: ParallelRankTopology,
    execution_identity: String,
    prototype: ForwardCaptureBundle,
    invocation_limits: (i32, i32),
    publication: eredu_runtime::CommunicationGroupRequirements,
}

impl PreparedTargetCapturePlacement {
    /// Shares source-free producer discovery between cold and bound construction.
    pub(crate) fn prepare(
        selected: &crate::SelectedExecution,
        descriptor: &eredu_core::ArchitectureDescriptor,
        execution_identity: &str,
        paths: &[String],
        component_layouts: impl FnOnce(
            usize,
        ) -> Result<
            Option<ComponentPartitionLayouts>,
            ComponentPartitionError,
        >,
    ) -> Result<Self, ComponentPartitionError> {
        let topology = selected.parallel_topology().ok_or_else(|| {
            CaptureError::Invalid("capture selection has no rank topology".into())
        })?;
        let element = match selected.partitioned_activation_dtype().ok_or_else(|| {
            CaptureError::Invalid("capture selection has no transport precision".into())
        })? {
            eredu_runtime::PipelineActivationDtype::Float16 => eredu_nn::TensorElementType::F16,
            eredu_runtime::PipelineActivationDtype::Bfloat16 => eredu_nn::TensorElementType::Bf16,
            eredu_runtime::PipelineActivationDtype::Float32 => eredu_nn::TensorElementType::F32,
            _ => {
                return Err(CaptureError::Unsupported(
                    "capture transport precision is unsupported".into(),
                )
                .into())
            }
        };
        let captures = paths
            .iter()
            .map(|path| {
                let point = descriptor
                    .observations
                    .get(path)
                    .ok_or_else(|| CaptureError::MissingPath(path.clone()))?;
                let width = match point.axes.as_deref() {
                    Some([batch, sequence, hidden])
                        if batch.dimension == Dim::Batch
                            && sequence.dimension == Dim::Sequence
                            && hidden.name == "hidden" =>
                    {
                        match hidden.dimension {
                            Dim::Known(width) if width > 0 => {
                                i32::try_from(width).map_err(|_| {
                                    CaptureError::Invalid("capture hidden width exceeds i32".into())
                                })?
                            }
                            _ => {
                                return Err(CaptureError::Unsupported(
                                    "capture hidden width is unresolved".into(),
                                ))
                            }
                        }
                    }
                    _ => {
                        return Err(CaptureError::Unsupported(
                            "target capture is not a complete decoder hidden sequence".into(),
                        ))
                    }
                };
                Ok((path.clone(), vec![1, 1, width]))
            })
            .collect::<Result<Vec<_>, CaptureError>>()?;
        let layouts = component_layouts(topology.world_size())?.ok_or_else(|| {
            CaptureError::Invalid("capture selection has no component placement".into())
        })?;
        let prototype = layouts.whole_tensor_capture_bundle(&captures, element)?;
        let mut elements = crate::linear_format::floating_compute_precisions(
            selected.text_realization().materialization_tasks(),
        )
        .into_iter()
        .filter_map(|dtype| match dtype {
            eredu_core::checkpoint::TensorDtype::F16 => Some(eredu_nn::TensorElementType::F16),
            eredu_core::checkpoint::TensorDtype::Bf16 => Some(eredu_nn::TensorElementType::Bf16),
            eredu_core::checkpoint::TensorDtype::F32 => Some(eredu_nn::TensorElementType::F32),
            _ => None,
        })
        .collect::<Vec<_>>();
        // A pipeline receiver can promote or narrow its residual stream to the
        // selected wire precision before the next producer executes.
        elements.push(element);
        let prototype = ForwardCaptureBundle::new(
            prototype
                .slots()
                .iter()
                .map(|slot| {
                    ForwardCaptureSlot::new(
                        slot.role().to_owned(),
                        slot.owner(),
                        slot.shape().to_vec(),
                        slot.placeholder_element(),
                    )
                    .and_then(|slot| slot.with_permitted_elements(elements.clone()))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(CaptureError::Invalid)?,
        )
        .map_err(CaptureError::Invalid)?;
        let group = selected.partitioned_session_group().ok_or_else(|| {
            CaptureError::Invalid("capture selection has no session publication group".into())
        })?;
        let publication = selected
            .communication_manifest()
            .and_then(|manifest| {
                manifest
                    .groups()
                    .iter()
                    .find(|candidate| candidate.id() == group)
            })
            .ok_or_else(|| {
                CaptureError::Invalid("capture selection has no publication manifest".into())
            })?
            .requirements()
            .clone();
        Ok(Self {
            topology,
            execution_identity: execution_identity.to_owned(),
            prototype,
            invocation_limits: selected.partitioned_invocation_limits().ok_or_else(|| {
                CaptureError::Invalid("capture selection has no invocation limits".into())
            })?,
            publication,
        })
    }
    /// Rank authority retained from the target selection, independent of native labels.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.topology
    }
    /// Identity of the retained execution whose declaration produced these sites.
    pub fn execution_identity(&self) -> &str {
        &self.execution_identity
    }
    /// Exact retained decoder invocation limits, independent of media route extents.
    pub const fn invocation_limits(&self) -> (i32, i32) {
        self.invocation_limits
    }
    /// Resolves only the admitted decoder batch and sequence; sites, widths, order,
    /// and transport precision remain the prepared architecture declaration.
    pub fn resolve(&self, decoder_shape: [u64; 2]) -> Result<ForwardCaptureBundle, CaptureError> {
        let batch = i32::try_from(decoder_shape[0])
            .map_err(|_| CaptureError::Invalid("capture batch exceeds i32".into()))?;
        let sequence = i32::try_from(decoder_shape[1])
            .map_err(|_| CaptureError::Invalid("capture sequence exceeds i32".into()))?;
        if batch <= 0
            || sequence <= 0
            || batch > self.invocation_limits.0
            || sequence > self.invocation_limits.1
        {
            return Err(CaptureError::Invalid(
                "target capture exceeds retained decoder invocation limits".into(),
            ));
        }
        let slots = self
            .prototype
            .slots()
            .iter()
            .map(|slot| {
                ForwardCaptureSlot::new(
                    slot.role().to_owned(),
                    slot.owner(),
                    vec![batch, sequence, slot.shape()[2]],
                    slot.placeholder_element(),
                )
                .and_then(|new| new.with_permitted_elements(slot.permitted_elements().to_vec()))
                .map_err(CaptureError::Invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bundle = ForwardCaptureBundle::new(slots).map_err(CaptureError::Invalid)?;
        self.publication
            .validate_requirement(
                &eredu_runtime::CommunicationOperationRequirement::failure_agreement(true),
            )
            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
        for slot in bundle.slots() {
            let mut dtypes = slot
                .permitted_elements()
                .iter()
                .map(|element| match element {
                    eredu_nn::TensorElementType::F16 => {
                        Ok(eredu_core::checkpoint::TensorDtype::F16)
                    }
                    eredu_nn::TensorElementType::Bf16 => {
                        Ok(eredu_core::checkpoint::TensorDtype::Bf16)
                    }
                    eredu_nn::TensorElementType::F32 => {
                        Ok(eredu_core::checkpoint::TensorDtype::F32)
                    }
                    eredu_nn::TensorElementType::F64 => {
                        Ok(eredu_core::checkpoint::TensorDtype::F64)
                    }
                    _ => Err(CaptureError::Unsupported(
                        "capture publication precision is unsupported".into(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            // The shared publication protocol announces the owner's actual
            // scalar type in one exact F32 header before allocating receivers.
            if slot.permitted_elements().len() > 1
                && !dtypes.contains(&eredu_core::checkpoint::TensorDtype::F32)
            {
                dtypes.push(eredu_core::checkpoint::TensorDtype::F32);
            }
            let elements = slot
                .shape()
                .iter()
                .try_fold(1usize, |total, dimension| {
                    usize::try_from(*dimension)
                        .ok()
                        .and_then(|dimension| total.checked_mul(dimension))
                })
                .ok_or_else(|| {
                    CaptureError::Invalid("capture publication extent overflows".into())
                })?;
            let limits = eredu_runtime::CommunicationTensorLimits::new(
                1,
                slot.shape().len(),
                elements,
                None,
            )
            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
            let requirement = eredu_runtime::CommunicationOperationRequirement::tensors(
                eredu_runtime::CommunicationOperation::Broadcast,
                dtypes,
                limits,
                true,
            )
            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
            self.publication
                .validate_requirement(&requirement)
                .map_err(|error| CaptureError::Invalid(error.to_string()))?;
        }
        Ok(bundle)
    }
}
