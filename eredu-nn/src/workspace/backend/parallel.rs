//! One rank's metadata execution of the ordinary distributed neural contracts.

use super::*;
use crate::{DistributedNeuralBackend, VocabularyParallelRange};
mod gather;

/// Validated rank identity for cold neural equation execution. This contains no
/// transport, group, device or native resource.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct WorkspaceParallelContext {
    rank: usize,
    partitions: usize,
}

impl WorkspaceParallelContext {
    /// Creates one rank's metadata context within a nonempty parallel group.
    pub fn new(rank: usize, partitions: usize) -> Result<Self, Error> {
        if partitions == 0 || rank >= partitions {
            return Err(Error::backend(
                "invalid workspace parallel rank or group size",
            ));
        }
        Ok(Self { rank, partitions })
    }

    /// This rank's coordinate in the group.
    pub const fn rank(self) -> usize {
        self.rank
    }

    /// Number of participating ranks.
    pub const fn size(self) -> usize {
        self.partitions
    }

    fn vocabulary_widths(
        self,
        range: &VocabularyParallelRange,
        context: &WorkspaceContext,
    ) -> Result<crate::BalancedVocabularyWidths, Error> {
        let controls = [
            std::mem::size_of::<crate::BalancedVocabularyWidths>(),
            std::mem::size_of::<crate::VocabularyRangeError>(),
            std::mem::size_of::<Result<crate::BalancedVocabularyWidths, crate::VocabularyRangeError>>(
            ),
            std::mem::size_of::<Result<crate::BalancedVocabularyWidths, Error>>(),
            std::mem::size_of::<(Self, &VocabularyParallelRange, &WorkspaceContext)>(),
            std::mem::size_of::<[usize; 2]>(),
        ];
        context.charge_metadata(
            controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        // Every admitted vocabulary owner must have at least one row. Reject
        // impossible cardinality before allocating a peer-width vector.
        if self.partitions > range.global_vocabulary {
            return Err(context.metadata_error(format_args!(
                "workspace vocabulary has more owners than rows"
            )));
        }
        range
            .balanced_peer_widths_plan(self.partitions, self.rank)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))
    }
}

impl DistributedNeuralBackend for WorkspaceBackend {
    fn vocabulary_parallel_embedding(
        mut spec: EmbeddingSpec,
        range: VocabularyParallelRange,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceEmbedding, Error> {
        range.validate_global_rows(spec.vocabulary)?;
        spec.vocabulary = i32::try_from(range.local.len()).map_err(|_| {
            context.metadata_error(format_args!(
                "workspace local vocabulary width overflows i32"
            ))
        })?;
        let mut embedding = Self::embedding(spec, context)?;
        embedding.projection.vocabulary_range = Some(range);
        Ok(embedding)
    }

    fn vocabulary_parallel_linear(
        mut spec: LinearSpec,
        range: VocabularyParallelRange,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceLinear, Error> {
        range.validate_global_rows(spec.output)?;
        spec.output = i32::try_from(range.local.len()).map_err(|_| {
            context.metadata_error(format_args!(
                "workspace local vocabulary width overflows i32"
            ))
        })?;
        let mut linear = Self::linear(spec, context)?;
        linear.vocabulary_range = Some(range);
        Ok(linear)
    }

    fn vocabulary_parallel_lookup(
        embedding: &mut WorkspaceEmbedding,
        input: &WorkspaceTensor,
        policy: EmbeddingLookupPolicy,
        parallel: &WorkspaceParallelContext,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        policy.validate()?;
        if !matches!(
            input.layout.dtype,
            WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
        ) {
            return Err(
                context.metadata_error(format_args!("workspace embedding requires integer IDs"))
            );
        }
        let projection = &embedding.projection;
        let range = projection.vocabulary_range.as_ref().ok_or_else(|| {
            context.metadata_error(format_args!(
                "workspace embedding has no vocabulary ownership"
            ))
        })?;
        parallel.vocabulary_widths(range, context)?;
        let mut shape = context.metadata_vec(
            input
                .shape()
                .len()
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        shape.extend_from_slice(input.shape());
        shape.push(projection.spec.input);
        let mut inputs = context.metadata_vec(
            projection
                .parameters
                .len()
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        inputs.push(input);
        inputs.extend(projection.parameters.iter().map(Parameter::as_ref));
        let contribution = WorkspaceTensor::operation(
            WorkspaceOperationKind::VocabularyParallelLookup {
                format: context.clone_metadata(&projection.spec.format)?,
                range: range.clone(),
                policy,
            },
            &inputs,
            &shape,
            WorkspaceDtype::Float32,
            context,
        )?;
        Self::sum_parallel(contribution, parallel, context)
    }

    fn vocabulary_parallel_project(
        linear: &mut WorkspaceLinear,
        input: &WorkspaceTensor,
        parallel: &WorkspaceParallelContext,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        project(linear, input, parallel, context, None)
    }

    fn vocabulary_parallel_project_with_input_observer(
        linear: &mut WorkspaceLinear,
        input: &WorkspaceTensor,
        parallel: &WorkspaceParallelContext,
        context: &WorkspaceContext,
        observer: Option<&mut dyn crate::ProjectionInputObserver<WorkspaceTensor>>,
    ) -> Result<WorkspaceTensor, Error> {
        project(linear, input, parallel, context, observer)
    }

    fn vocabulary_parallel_embedding_project(
        embedding: &mut WorkspaceEmbedding,
        input: &WorkspaceTensor,
        parallel: &WorkspaceParallelContext,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        project(&mut embedding.projection, input, parallel, context, None)
    }

    fn vocabulary_parallel_embedding_project_with_input_observer(
        embedding: &mut WorkspaceEmbedding,
        input: &WorkspaceTensor,
        parallel: &WorkspaceParallelContext,
        context: &WorkspaceContext,
        observer: Option<&mut dyn crate::ProjectionInputObserver<WorkspaceTensor>>,
    ) -> Result<WorkspaceTensor, Error> {
        project(
            &mut embedding.projection,
            input,
            parallel,
            context,
            observer,
        )
    }

    fn sum_parallel(
        value: WorkspaceTensor,
        parallel: &WorkspaceParallelContext,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        WorkspaceTensor::operation(
            WorkspaceOperationKind::Collective(WorkspaceCollective::Sum {
                partitions: parallel.size(),
                rank: parallel.rank(),
            }),
            &[&value],
            value.shape(),
            value.layout.dtype,
            context,
        )
    }
}

fn project(
    linear: &mut WorkspaceLinear,
    input: &WorkspaceTensor,
    parallel: &WorkspaceParallelContext,
    context: &WorkspaceContext,
    observer: Option<&mut dyn crate::ProjectionInputObserver<WorkspaceTensor>>,
) -> Result<WorkspaceTensor, Error> {
    let range = linear.vocabulary_range.as_ref().ok_or_else(|| {
        context.metadata_error(format_args!(
            "workspace projection has no vocabulary ownership"
        ))
    })?;
    let widths = parallel.vocabulary_widths(range, context)?;
    let mut peer_widths = context.metadata_vec(widths.len())?;
    peer_widths.extend(widths.widths());
    let local = linear.forward_with_input_observer(input, context, observer)?;
    let axis = local.shape().len().checked_sub(1).ok_or_else(||
        context.metadata_error(format_args!("workspace vocabulary projection returned a scalar")))?;
    crate::gather_uneven_axis(&gather::Operations(context), &local, axis, parallel.rank(), &peer_widths)
}

impl WorkspaceTensor {
    /// Records the same uneven-axis gathering equations used by vocabulary
    /// projection. Runtime communication passes the selected group rank/widths.
    pub fn gather_uneven_axis(&self, axis: usize, rank: usize, widths: &[usize],
        context: &WorkspaceContext) -> Result<Self, Error> {
        crate::gather_uneven_axis(&gather::Operations(context), self, axis, rank, widths)
    }
}
