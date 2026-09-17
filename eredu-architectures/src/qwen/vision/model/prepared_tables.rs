//! Actual tower execution from the original B owner's fixed source tables.
use super::*;
use eredu_nn::{
    multimodal::multi_axis_rotary_embeddings_prepared,
    sequence_layout::{PatchEncoderTables, PatchPositionTableSpec},
};

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionStatic<B> {
    fn checked_source_tables<'a>(
        &self,
        source: &'a PreparedModelInputOwner<B::Tensor>,
        sequence: i32,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<PatchEncoderTables<'a>, Error> {
        metadata.controls::<PatchEncoderTables<'_>>()?;
        let tables = source.original_encoder_tables().ok_or_else(|| {
            metadata.error(format_args!(
                "prepared encoder source has no original tables"
            ))
        })?;
        self.checked_table_view(tables, sequence, metadata)
    }
    fn checked_table_view<'a>(
        &self,
        tables: PatchEncoderTables<'a>,
        sequence: i32,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<PatchEncoderTables<'a>, Error> {
        metadata.controls::<PatchEncoderTables<'_>>()?;
        if tables.layout().patches() != sequence
            || tables.layout().spec()
                != super::super::encoder_table_spec(&self.config).map_err(|cause| match metadata
                    .context()
                {
                    Some(context) => context.metadata_source(cause),
                    None => Error::backend_source(cause),
                })?
        {
            return Err(metadata.error(format_args!(
                "prepared encoder tables differ from the selected tower"
            )));
        }
        Ok(tables)
    }

    /// Executes the existing patch/position equation with real original tables.
    /// Native tensors and private work remain ordinary execution in this unit.
    pub(crate) fn begin_with_original_tables(
        &mut self,
        pixels: &B::Tensor,
        source: &PreparedModelInputOwner<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        let metadata = crate::decoder::identity::Metadata::new(
            metadata.or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(VisionStateTables<B::Tensor>, VisionState<B::Tensor>)>()?;
        let tables = self.checked_source_tables(source, pixels.dim(0), metadata)?;
        self.begin_using_tables(
            pixels,
            tables,
            VisionStateTables::Original(source.clone()),
            context,
            metadata,
        )
    }
    pub(crate) fn begin_with_projected_tables(
        &mut self,
        pixels: &B::Tensor,
        source: &eredu_runtime::input::OriginalEncoderTableProjection,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(VisionStateTables<B::Tensor>, VisionState<B::Tensor>)>()?;
        let tables = self.checked_table_view(source.tables(), pixels.dim(0), metadata)?;
        self.begin_using_tables(
            pixels,
            tables,
            VisionStateTables::Projected(source.clone()),
            context,
            metadata,
        )
    }
    fn begin_using_tables(
        &mut self,
        pixels: &B::Tensor,
        tables: PatchEncoderTables<'_>,
        retained: VisionStateTables<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        metadata.controls::<(B::Tensor, PatchEncoderTables<'_>, VisionState<B::Tensor>)>()?;
        let hidden = self.patch.forward(pixels, context)?;
        let positions = learned_positions::<B>(&mut self.position, &tables, context)?;
        let hidden = hidden.add(&positions, context)?;
        let sequence = hidden.dim(0);
        let (hidden, state) = self.reorder_continuation(
            sequence,
            Some(hidden),
            retained,
            || rotary::<B::Tensor>(&tables, context),
            context,
            metadata.context(),
        )?;
        Ok((hidden.expect("prepared patch input supplied"), state))
    }

    /// Receivers reuse the exact fixed tables without patch/position projection.
    pub(crate) fn continuation_state_with_original_tables(
        &self,
        source: &PreparedModelInputOwner<B::Tensor>,
        sequence: i32,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<VisionState<B::Tensor>, Error> {
        let metadata = crate::decoder::identity::Metadata::new(
            metadata.or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(VisionStateTables<B::Tensor>, VisionState<B::Tensor>)>()?;
        let tables = self.checked_source_tables(source, sequence, metadata)?;
        self.reorder_continuation(
            sequence,
            None,
            VisionStateTables::Original(source.clone()),
            || rotary::<B::Tensor>(&tables, context),
            context,
            metadata.context(),
        )
        .map(|(_, state)| state)
    }
}

fn learned_positions<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    embedding: &mut B::Embedding,
    tables: &PatchEncoderTables<'_>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let length = tables.layout().patches();
    let indices = tables.learned_indices();
    match tables.layout().spec().positions {
        PatchPositionTableSpec::Gathered { .. } => embedding.forward(
            &B::Tensor::from_i32_slice(indices, &[length], context)?,
            context,
        ),
        PatchPositionTableSpec::Interpolated { .. } => {
            let weights = tables.learned_weights();
            let n = length as usize;
            project_interpolated_positions::<B>(embedding, context, |corner| {
                let range = corner * n..(corner + 1) * n;
                Ok((
                    B::Tensor::from_i32_slice(&indices[range.clone()], &[length], context)?,
                    B::Tensor::from_f32_slice(&weights[range], &[length, 1], context)?,
                ))
            })
        }
    }
}
fn rotary<T: Tensor>(
    tables: &PatchEncoderTables<'_>,
    context: &T::Context,
) -> Result<(T, T), Error> {
    let positions = T::from_i32_slice(
        tables.spatial_positions(),
        &[tables.layout().patches(), 2],
        context,
    )?;
    multi_axis_rotary_embeddings_prepared(&positions, tables.rotary(), context)
}
