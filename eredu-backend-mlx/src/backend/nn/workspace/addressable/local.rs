//! Descriptive receive-row reconstruction shared by indexed source modes.
use super::*;

pub(super) struct LocalAddressableSource<'a> {
    operation: WorkspaceOperationView<'a>,
    region: &'a WorkspaceExpertRegion,
    pub(super) source: WorkspaceAddressableRegionView<'a>,
    pub(super) maximum: usize,
}
impl<'a> LocalAddressableSource<'a> {
    pub(super) fn prepare(
        operation: WorkspaceOperationView<'a>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(size_of::<(
            Self,
            WorkspaceOperationView<'_>,
            &WorkspaceContext,
            Result<Self, Error>,
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "local indexed source differs from its retained expert region"
            ))
        };
        let WorkspaceOperationKindView::ExpertRegion(region) = operation.kind else {
            return Err(invalid());
        };
        let view = region.as_view();
        view.validate()?;
        let shape = ExpertRegionInputShape::inspect(
            operation.inputs.get(0).ok_or_else(invalid)?.shape(),
            operation.inputs.get(1).ok_or_else(invalid)?.shape(),
        )?;
        if usize::try_from(shape.rows).ok() != Some(view.source_rows)
            || usize::try_from(shape.routes).ok() != Some(view.routes_per_row)
            || shape.width != view.kernel.dimensions().0
        {
            return Err(invalid());
        }
        let source = view.addressable.ok_or_else(invalid)?;
        let maximum = view.maximum_received_rows().ok_or_else(invalid)?;
        if maximum == 0 || source.chunks.rows != maximum || operation.inputs.len() != 4 {
            return Err(invalid());
        }
        Ok(Self {
            operation,
            region,
            source,
            maximum,
        })
    }
    pub(super) fn report(
        &self,
        rows: usize,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTraceReport, Error> {
        context.charge_metadata(size_of::<(
            &Self,
            usize,
            &WorkspaceContext,
            WorkspaceAddressableRegionView<'_>,
            eredu_nn::GroupSelection<WorkspaceTensor>,
            WorkspaceTraceReport,
            Result<WorkspaceTraceReport, Error>,
            Vec<WorkspaceTensor>,
            [i32; 2],
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "local indexed rows exceed their retained source"
            ))
        };
        if rows == 0 || rows > self.maximum {
            return Err(invalid());
        }
        let mut source = self.source;
        source.chunks.rows = rows;
        let mut inputs = context.metadata_vec(4)?;
        for (index, input) in self.operation.inputs.iter().enumerate() {
            let shape = [
                i32::try_from(rows).map_err(|_| invalid())?,
                if index == 0 {
                    source.kernel.dimensions().0
                } else {
                    1
                },
            ];
            inputs.push(WorkspaceTensor::existing(
                context
                    .layout(
                        &shape,
                        if index == 1 {
                            WorkspaceDtype::Int32
                        } else {
                            input.dtype()
                        },
                    )?
                    .with_representation(input.representation()),
                &context,
            )?);
        }
        let routes =
            eredu_nn::GroupSelection::new(inputs[1].clone(), inputs[2].clone(), inputs[3].clone());
        let mut observe = |_: eredu_nn::workspace::WorkspaceAddressableObservationView<'_>| {
            let value = self.region.observation().ok_or_else(invalid)?;
            Ok(eredu_nn::workspace::WorkspaceAddressableObservationSource {
                before: value.before,
                after: value.after,
                unit_dtype: value.unit_dtype,
            })
        };
        context.begin_span();
        let output = eredu_nn::workspace::record_addressable_region_with_observation(
            source,
            &inputs[0],
            &routes,
            &context,
            if self.region.observation().is_some() {
                Some(&mut observe)
            } else {
                None
            },
        )?;
        let (value, bias) = output.into_parts();
        let mut outputs = context.metadata_vec(1 + usize::from(bias.is_some()))?;
        outputs.push(value);
        outputs.extend(bias);
        context.finish_report(&outputs)
    }
}
