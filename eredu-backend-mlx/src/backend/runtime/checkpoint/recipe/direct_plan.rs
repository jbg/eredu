//! Owned, reusable admission of the exact direct-read dispatch.

use std::collections::{BTreeMap, BTreeSet};

use eredu_runtime::{working_memory::WorkingMemoryError, WeightBinding};

use super::*;
use crate::backend::nn::workspace::NativeAllocationFacts;

#[derive(Debug, thiserror::Error)]
pub(crate) enum PreparedDirectReadError {
    #[error("direct read plan is unavailable: {reason}")]
    Unpriced {
        reason: String,
        #[source]
        source: WorkingMemoryError,
    },
    #[error("direct read planning or execution failed: {0}")]
    Recipe(#[from] WeightRecipeError),
}

impl PreparedDirectReadError {
    fn unpriced(reason: impl Into<String>) -> Self {
        Self::Unpriced {
            reason: reason.into(),
            source: WorkingMemoryError::UnknownBound,
        }
    }
}

/// One independently allocated owner output. Alias bindings belong to the
/// caller's canonical-owner dispatch and must not enter this read plan.
#[derive(Debug, Clone)]
pub(crate) struct DirectReadOutput {
    binding: WeightBinding,
    shape: Vec<i32>,
    dtype: Dtype,
}

impl DirectReadOutput {
    pub(crate) fn binding(&self) -> &WeightBinding {
        &self.binding
    }
    pub(crate) fn name(&self) -> &str {
        self.binding.name()
    }
    pub(crate) fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub(crate) fn logical_bytes(&self) -> u64 {
        self.binding.expected_bytes()
    }
    pub(crate) fn capacity_bytes(
        &self,
        allocation: NativeAllocationFacts,
    ) -> Result<u64, PreparedDirectReadError> {
        super::workspace::direct_output_capacity(&self.binding, self.logical_bytes(), allocation)
            .map_err(Into::into)
    }
}

#[derive(Clone)]
struct DirectBatch {
    names: Vec<String>,
    reads: Vec<DirectRecipeRead>,
}

/// Contains immutable admitted-file/read metadata, never a tensor payload.
/// Preparation may admit source headers and update metadata telemetry, but
/// performs no payload reads or native allocation. Pricing and cloning are
/// inert after preparation. Reuse
/// consumes clones of the same admitted spans; it cannot select an ordinary
/// recipe, reopen admission, or discover a different checkpoint source.
#[derive(Clone)]
pub(crate) struct PreparedDirectReadPlan {
    batches: Vec<DirectBatch>,
    outputs: Vec<DirectReadOutput>,
}

impl std::fmt::Debug for PreparedDirectReadPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedDirectReadPlan")
            .field("batches", &self.batches.len())
            .field("outputs", &self.outputs)
            .finish()
    }
}

impl PreparedDirectReadPlan {
    pub(crate) fn prepare(
        source: &dyn CheckpointSource,
        bindings: &[WeightBinding],
    ) -> Result<Self, PreparedDirectReadError> {
        let mut names = BTreeSet::new();
        for binding in bindings {
            if binding.is_alias() {
                return Err(PreparedDirectReadError::unpriced(format!(
                    "alias {} requires an explicit canonical owner",
                    binding.name()
                )));
            }
            if !names.insert(binding.name()) {
                return Err(
                    WeightRecipeError::Preflight("duplicate direct output name".into()).into(),
                );
            }
        }
        let mut result = Self {
            batches: Vec::new(),
            outputs: Vec::new(),
        };
        for batch in plan_binding_reads(source, bindings)? {
            let (bindings, reads) = match batch {
                BindingReadBatch::Direct { bindings, reads } => (bindings, reads),
                BindingReadBatch::Ordinary(binding) => {
                    return Err(PreparedDirectReadError::unpriced(format!(
                        "binding {} requires an unpriced ordinary materializer",
                        binding.name()
                    )))
                }
            };
            if bindings.len() != reads.len() {
                return Err(WeightRecipeError::Preflight(
                    "direct batch output count mismatch".into(),
                )
                .into());
            }
            for (binding, read) in bindings.iter().zip(&reads) {
                super::workspace::validate_direct_output(binding, read.read.output().byte_len)?;
                result.outputs.push(DirectReadOutput {
                    binding: (*binding).clone(),
                    shape: read.shape.clone(),
                    dtype: read.dtype,
                });
            }
            result.batches.push(DirectBatch {
                names: bindings
                    .iter()
                    .map(|binding| binding.name().to_owned())
                    .collect(),
                reads,
            });
        }
        Ok(result)
    }

    pub(crate) fn outputs(&self) -> &[DirectReadOutput] {
        &self.outputs
    }
    pub(crate) fn workspace_bound(
        &self,
        allocation: NativeAllocationFacts,
    ) -> Result<eredu_core::WorkspaceBound, PreparedDirectReadError> {
        super::workspace::direct_workspace_bound(
            self.outputs
                .iter()
                .map(|output| (output.binding(), output.logical_bytes())),
            allocation,
        )
        .map_err(Into::into)
    }
    pub(crate) fn host_staging_bytes(&self) -> u64 {
        0
    }

    /// The caller must arm native recovery before entry and retain the supplied
    /// arrays until exact transfer completion (including failure). Every input
    /// in a batch enters that custody before its first fallible stream copy.
    /// Direct initialization failures expose no partially initialized array.
    pub(crate) fn materialize(
        &self,
        source_stream: &Stream,
        execution_stream: &Stream,
        retain: impl FnMut(&Array),
    ) -> Result<BTreeMap<String, Array>, PreparedDirectReadError> {
        let mut outputs = BTreeMap::new();
        self.materialize_into(
            source_stream,
            execution_stream,
            retain,
            |name, output| -> Result<(), PreparedDirectReadError> {
                outputs.insert(name.to_owned(), output);
                Ok(())
            },
        )?;
        Ok(outputs)
    }

    /// Same admitted read/copy/retention worker with a borrowed output name.
    /// The sink owns its actual destination and any refusal. Every output has
    /// entered recovery custody before the sink can fail; no temporary named
    /// map or cloned output name is needed by a prepared caller.
    pub(crate) fn materialize_into<E: From<PreparedDirectReadError>>(
        &self,
        source_stream: &Stream,
        execution_stream: &Stream,
        mut retain: impl FnMut(&Array),
        mut publish: impl FnMut(&str, Array) -> Result<(), E>,
    ) -> Result<(), E> {
        if execution_stream
            .get_device()
            .and_then(|device| device.get_type())
            .map_err(WeightRecipeError::Mlx)
            .map_err(PreparedDirectReadError::from)
            .map_err(E::from)?
            != safemlx::DeviceType::Gpu
        {
            return Err(PreparedDirectReadError::unpriced(
                "execution stream is not selected Metal",
            )
            .into());
        }
        for batch in &self.batches {
            let inputs = DirectRecipeRead::materialize_many(batch.reads.clone())
                .map_err(PreparedDirectReadError::from)
                .map_err(E::from)?;
            for input in &inputs {
                retain(input);
            }
            for (name, input) in batch.names.iter().zip(inputs) {
                let output = if source_stream == execution_stream {
                    input
                } else {
                    input
                        .copy(execution_stream)
                        .map_err(WeightRecipeError::Mlx)
                        .map_err(PreparedDirectReadError::from)
                        .map_err(E::from)?
                };
                retain(&output);
                publish(name, output)?;
            }
        }
        Ok(())
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
