//! Output metadata and exact source occurrences for a native recipe producer.
//! Leaf ranges describe real detached reads; the recipe output is never an
//! encoded read merely because its shape matches one of its inputs.
use super::*;
use crate::backend::{
    nn::workspace::{
        NativeAllocationFacts, OrdinaryNativeControls, ResidentExecutionMechanisms,
        SpeculativeNumericalRecipe,
    },
    runtime::checkpoint::recipe::{mlx_dtype, trace_ordinary_recipe},
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeMetadata},
    store::{CheckpointSource, RetainedCheckpointSource, TensorSelection},
};
use eredu_nn::Tensor;
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage,
    WorkspaceFloatingType, WorkspaceMetadataError, WorkspaceRepresentation, WorkspaceTensor,
};

mod cold;
type ReadCustody = eredu_runtime::working_memory::CompiledRecipeCustody<
    eredu_runtime::working_memory::SharedNativeInitializationCustody,
>;
pub(in crate::backend::runtime::residency::manager::construction) struct PreparedLeafRead {
    read: eredu_checkpoint::recipe::EncodedRecipeRead<ReadCustody>,
    shape: Vec<i32>,
    dtype: safemlx::Dtype,
}
impl PreparedLeafRead {
    pub(in crate::backend::runtime::residency::manager::construction) fn encoded(
        &self,
    ) -> &eredu_checkpoint::recipe::EncodedRecipeRead<ReadCustody> {
        &self.read
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn dtype(
        &self,
    ) -> safemlx::Dtype {
        self.dtype
    }
}

pub(in crate::backend::runtime::residency::manager::construction) struct PreparedRecipeLeaf {
    pub(in crate::backend::runtime::residency::manager::construction) key: String,
    pub(in crate::backend::runtime::residency::manager::construction) selection: TensorSelection,
    pub(in crate::backend::runtime::residency::manager::construction) read: PreparedLeafRead,
}

pub(in crate::backend::runtime::residency::manager::construction) struct MaterializedReadPlan {
    pub(in crate::backend::runtime::residency::manager::construction) recipe: DerivedWeightRecipe,
    pub(in crate::backend::runtime::residency::manager::construction) output: RecipeMetadata,
    pub(in crate::backend::runtime::residency::manager::construction) shape: Vec<i32>,
    pub(in crate::backend::runtime::residency::manager::construction) dtype: safemlx::Dtype,
    pub(in crate::backend::runtime::residency::manager::construction) leaves:
        Vec<PreparedRecipeLeaf>,
    pub(in crate::backend::runtime::residency::manager::construction) numerical:
        SpeculativeNumericalRecipe,
    pub(in crate::backend::runtime::residency::manager::construction) original_numerical:
        SpeculativeNumericalRecipe,
    pub(in crate::backend::runtime::residency::manager::construction) validations: usize,
    pub(in crate::backend::runtime::residency::manager::construction) wrapper_metadata: u64,
    pub(in crate::backend::runtime::residency::manager::construction) wrapper_observed:
        OrdinaryNativeControls,
    _funding: HostMetadataFunding,
}

/// Unique cold growth authority remains outside the shared immutable source.
/// Descriptors alias only `source`; they cannot reopen its metadata account.
pub(in crate::backend::runtime::residency::manager::construction) struct PreparedMaterializedRead {
    source: std::sync::Arc<MaterializedReadPlan>,
    preparation: Option<eredu_runtime::working_memory::PreparedConstructionMetadata>,
}
impl std::ops::Deref for PreparedMaterializedRead {
    type Target = MaterializedReadPlan;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}
impl PreparedMaterializedRead {
    pub(in crate::backend::runtime::residency::manager::construction) fn shared(
        &self,
    ) -> std::sync::Arc<MaterializedReadPlan> {
        self.source.clone()
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn seal_metadata(
        &mut self,
    ) -> Result<(), eredu_nn::workspace::HostMetadataFundingError> {
        if let Some(preparation) = self.preparation.take() {
            // Every alias already owns this same accounting custody.
            drop(preparation.seal()?);
        }
        Ok(())
    }
}

impl MaterializedReadPlan {
    pub(in crate::backend::runtime::residency::manager::construction) fn funding(
        &self,
    ) -> &HostMetadataFunding {
        &self._funding
    }
    /// The caller owns this cold plan and its metadata context. The imported
    /// leaves describe the independently quoted Host-copy destinations; they
    /// grant neither existing-storage credit nor payload access.
    pub(in crate::backend::runtime::residency::manager::construction) fn prepare(
        recipe: &DerivedWeightRecipe,
        source: &RetainedCheckpointSource,
        pool: &MemoryLedger,
        preparation: eredu_runtime::working_memory::PreparedConstructionMetadata,
        mechanism: ResidentExecutionMechanisms,
    ) -> Result<PreparedMaterializedRead, WeightRecipeError> {
        let funding = preparation.funding().clone();
        let original_mechanism = mechanism;
        let mechanism = mechanism.ordinary_storage();
        let owned_context = mechanism
            .context(funding.clone())
            .map_err(|cause| WeightRecipeError::Workspace(cause.into()))?;
        let context = &owned_context;
        let allocation = mechanism.allocation();
        let recipe = cold::clone_recipe(recipe, context)?;
        let output = cold::infer(&recipe, source.as_ref(), context)?;
        let dtype = mlx_dtype(&output.dtype)?;
        let mut shape = context.metadata_vec(output.shape.len())?;
        for &n in &output.shape {
            shape.push(
                i32::try_from(n).map_err(|_| {
                    WeightRecipeError::ArithmeticOverflow("recipe output dimension")
                })?,
            );
        }
        let mut leaves = context.metadata_vec(0)?;
        let equation =
            trace_ordinary_recipe(&recipe, context, |key, selection, context| {
                let leaf_recipe = DerivedWeightRecipe::Source {
                    key: context.metadata_string(format_args!("{key}"))?,
                    selection: clone_selection(selection, context)?,
                };
                let read = pool
                    .prepare_encoded_recipe(source, &leaf_recipe)
                    .map_err(|cause| WeightRecipeError::Workspace(context.metadata_source(cause)))?
                    .ok_or_else(|| {
                        WeightRecipeError::Workspace(WorkspaceMetadataError::Unqualified.into())
                    })?;
                let dtype = mlx_dtype(&read.output().dtype)?;
                let mut shape = context.metadata_vec(read.output().shape.len())?;
                for &dimension in &read.output().shape {
                    shape.push(i32::try_from(dimension).map_err(|_| {
                        WeightRecipeError::ArithmeticOverflow("recipe leaf dimension")
                    })?);
                }
                let read = PreparedLeafRead { read, shape, dtype };
                let value = leaf_value(&read, context, allocation)?;
                context.reserve_metadata_vec(&mut leaves, 1)?;
                let DerivedWeightRecipe::Source { key, selection } = leaf_recipe else {
                    unreachable!("constructed Source recipe")
                };
                leaves.push(PreparedRecipeLeaf {
                    key,
                    selection,
                    read,
                });
                Ok(value)
            })?;
        if equation.output.shape() != shape || leaves.len() != equation.sources.len() {
            return Err(WeightRecipeError::ShapeMismatch);
        }
        let validations = equation.validations.len();
        let (wrapper_metadata, wrapper_observed) = equation.ordinary_wrapper_population();
        let numerical = equation.finish_native_population(mechanism, context)?;
        // The second source profile uses the actual Original mechanism over
        // the same equation and genuine leaf occurrences. Ordinary allocation
        // facts are never reinterpreted as prepared-arena authority.
        let original_context = original_mechanism
            .context(funding.clone())
            .map_err(|cause| WeightRecipeError::Workspace(cause.into()))?;
        let mut ordinal = 0usize;
        let original_equation =
            trace_ordinary_recipe(&recipe, &original_context, |key, selection, context| {
                let leaf = leaves
                    .get(ordinal)
                    .ok_or(WeightRecipeError::ShapeMismatch)?;
                if leaf.key != key || &leaf.selection != selection {
                    return Err(WeightRecipeError::ShapeMismatch);
                }
                ordinal = ordinal
                    .checked_add(1)
                    .ok_or(WeightRecipeError::ArithmeticOverflow(
                        "recipe source occurrence",
                    ))?;
                leaf_value(&leaf.read, context, original_mechanism.allocation())
            })?;
        if ordinal != leaves.len()
            || original_equation.output.shape() != shape
            || original_equation.validations.len() != validations
        {
            return Err(WeightRecipeError::ShapeMismatch);
        }
        let original_numerical =
            original_equation.finish_native_population(original_mechanism, &original_context)?;
        let mut transfers = original_context.metadata_vec(leaves.len())?;
        for leaf in &leaves {
            transfers.push((
                leaf.read.shape().len(),
                leaf.read.dtype(),
                original_mechanism
                    .allocation()
                    .fixed_buffer_capacity(leaf.read.encoded().output().byte_len())
                    .map_err(|cause| {
                        WeightRecipeError::Workspace(original_context.metadata_source(cause))
                    })?,
            ));
        }
        let output_capacity = original_mechanism
            .allocation()
            .fixed_buffer_capacity(output.byte_len())
            .map_err(|cause| {
                WeightRecipeError::Workspace(original_context.metadata_source(cause))
            })?;
        let original_numerical = original_numerical
            .with_prepared_recipe_transfers(
                transfers.into_iter(),
                shape.len(),
                dtype,
                output_capacity,
                &original_context,
            )
            .map_err(|cause| {
                WeightRecipeError::Workspace(original_context.metadata_source(cause))
            })?;
        let value = Self {
            recipe,
            output,
            shape,
            dtype,
            leaves,
            numerical,
            original_numerical,
            validations,
            wrapper_metadata,
            wrapper_observed,
            _funding: funding,
        };
        original_context
            .charge_metadata(std::mem::size_of::<(
                PreparedMaterializedRead,
                Result<PreparedMaterializedRead, WeightRecipeError>,
            )>())
            .map_err(|cause| WeightRecipeError::Workspace(cause.into()))?;
        let source = original_context
            .metadata_arc(value)
            .map_err(|cause| WeightRecipeError::Workspace(cause.into()))?;
        Ok(PreparedMaterializedRead {
            source,
            preparation: Some(preparation),
        })
    }

    /// Exact physical source requests, in the same order as the native source
    /// callback. Repeated keys remain distinct occurrences and retain selections.
    pub(in crate::backend::runtime::residency::manager::construction) fn encoded_leaves(
        &self,
    ) -> impl ExactSizeIterator<Item = eredu_checkpoint::recipe::EncodedRecipeReadView<'_>> {
        self.leaves
            .iter()
            .map(|leaf| leaf.read.encoded().borrowed())
    }
}

fn clone_selection(
    selection: &TensorSelection,
    context: &WorkspaceContext,
) -> Result<TensorSelection, WeightRecipeError> {
    Ok(match selection {
        TensorSelection::Indices { axis, indices } => {
            let mut copy = context.metadata_vec(indices.len())?;
            copy.extend_from_slice(indices);
            TensorSelection::Indices {
                axis: *axis,
                indices: copy,
            }
        }
        TensorSelection::Contiguous {
            offset_elements,
            shape,
        } => {
            let mut copy = context.metadata_vec(shape.len())?;
            copy.extend_from_slice(shape);
            TensorSelection::Contiguous {
                offset_elements: *offset_elements,
                shape: copy,
            }
        }
        TensorSelection::Range { axis, start, end } => TensorSelection::Range {
            axis: *axis,
            start: *start,
            end: *end,
        },
        TensorSelection::Full => TensorSelection::Full,
    })
}

fn leaf_value(
    read: &PreparedLeafRead,
    context: &WorkspaceContext,
    allocation: NativeAllocationFacts,
) -> Result<WorkspaceTensor, WeightRecipeError> {
    use safemlx::Dtype as N;
    let (logical, floating) = match read.dtype() {
        N::Float32 => (
            WorkspaceDtype::Float32,
            Some(WorkspaceFloatingType::Float32),
        ),
        N::Float16 => (
            WorkspaceDtype::Float32,
            Some(WorkspaceFloatingType::Float16),
        ),
        N::Bfloat16 => (
            WorkspaceDtype::Float32,
            Some(WorkspaceFloatingType::Bfloat16),
        ),
        N::Int32 => (WorkspaceDtype::Int32, None),
        N::Uint32 => (WorkspaceDtype::Uint32, None),
        N::Uint8 => (WorkspaceDtype::Uint8, None),
        N::Bool => (WorkspaceDtype::Bool, None),
        _ => {
            return Err(WeightRecipeError::Workspace(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
    };
    let capacity = allocation
        .fixed_buffer_capacity(read.encoded().output().byte_len())
        .map_err(|cause| WeightRecipeError::Workspace(context.metadata_source(cause)))?;
    let storage = WorkspaceExistingStorage::try_new(Some(capacity), context)?;
    let layout = context
        .layout(read.shape(), logical)?
        .with_representation(floating.map(|dtype| WorkspaceRepresentation::new(dtype, true)));
    Ok(WorkspaceTensor::existing_with_storage(
        layout, &storage, context,
    )?)
}
