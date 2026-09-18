//! Actual selected compact rows followed by the ordinary grouped equation.
use super::*;
use eredu_nn::GroupSelection;

pub(crate) struct AddressableChildSource {
    pub(crate) report: WorkspaceTraceReport,
    pub(crate) output_layouts: Vec<WorkspaceLayout>,
    pub(crate) capture:crate::backend::array_copy::CaptureNativePopulation,
    pub(crate) unit_representation:Option<WorkspaceRepresentation>,
}
impl AddressableChildSource {
    /// Parameter rows are member-major, in the exact physical binder order
    /// within each member. Each source already describes one actual group.
    /// Replacement slicing is a separate source preceding these row frontiers.
    pub(crate) fn prepare(
        source: WorkspaceAddressableRegionView<'_>,
        inputs: &[WorkspaceLayout],
        members: usize,
        parameters: &[WorkspaceLayout],
        mechanism: ResidentExecutionMechanisms,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::prepare_with_observation(source,inputs,members,parameters,mechanism,funding,None,None)
    }
    pub(crate) fn prepare_with_observation(
        source:WorkspaceAddressableRegionView<'_>,inputs:&[WorkspaceLayout],members:usize,
        parameters:&[WorkspaceLayout],mechanism:ResidentExecutionMechanisms,funding:&HostMetadataFunding,
        observation:Option<WorkspaceAddressableObservationSource>,source_groups:Option<&WorkspaceLayout>,
    )->Result<Self,Error>{
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())?;
        context.charge_metadata(size_of::<(
            Self,
            WorkspaceContext,
            WorkspaceGroupedBank,
            Vec<WorkspaceTensor>,
            Vec<&WorkspaceTensor>,
            Vec<WorkspaceLayout>,
            GroupSelection<WorkspaceTensor>,
            Result<Self, Error>,
            [usize; 4],
        )>())?;
        source.validate()?;
        let invalid = || {
            context.metadata_error(format_args!(
                "addressable child differs from selected compact parameter rows"
            ))
        };
        if inputs.len() != 4
            || members == 0
            || members > source.chunks.members
            || parameters.is_empty()
            || !parameters.len().is_multiple_of(members)
        {
            return Err(invalid());
        }
        let rows = inputs[0].shape().first().copied().ok_or_else(invalid)?;
        let (width, _) = source.kernel.dimensions();
        let routes = i32::try_from(source.chunks.routes).map_err(|_| invalid())?;
        if rows <= 0
            || rows as usize > source.chunks.chunk_rows
            || inputs[0].shape() != [rows, width]
            || inputs[1].shape() != [rows, routes]
            || inputs[2].shape() != inputs[1].shape()
            || inputs[3].shape() != inputs[1].shape()
            || !matches!(
                inputs[1].dtype(),
                WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
            )
        {
            return Err(invalid());
        }
        let bank = source
            .kernel
            .compact(i32::try_from(members).map_err(|_| invalid())?, &context)?;
        let fields = parameters.len() / members;
        let mut values = context.metadata_vec(4)?;
        for layout in inputs {
            values.push(WorkspaceTensor::existing(
                context
                    .layout(layout.shape(), layout.dtype())?
                    .with_representation(layout.representation()),
                &context,
            )?);
        }
        let mut parameter_rows = context.metadata_vec(parameters.len())?;
        for layout in parameters {
            if layout.shape().first() != Some(&1) {
                return Err(invalid());
            }
            parameter_rows.push(WorkspaceTensor::existing(
                context
                    .layout(layout.shape(), layout.dtype())?
                    .with_representation(layout.representation()),
                &context,
            )?);
        }
        context.begin_span();
        let mut compact = context.metadata_vec(fields)?;
        let mut rows = context.metadata_vec(members)?;
        for field in 0..fields {
            rows.clear();
            for member in 0..members {
                rows.push(parameter_rows[member * fields + field].clone());
            }
            // The native binder always invokes concatenate, including a single
            // selected member; its alias/copy behavior stays in the usual facts.
            compact.push(WorkspaceTensor::concatenate(&rows, 0, &context)?);
        }
        let mut borrowed = context.metadata_vec(fields)?;
        borrowed.extend(compact.iter());
        let selection =
            GroupSelection::new(values[1].clone(), values[2].clone(), values[3].clone());
        let mut observed=observation.map(|descriptor|{
            let retained=super::super::parallel::ExpertLocalObservationSource::from_addressable(descriptor)
                .ok_or_else(invalid)?;
            let mut observer=super::super::parallel::GroupedSourceObserver::new(retained,&values[0],&context)?;
            let groups=source_groups.ok_or_else(invalid)?;
            if groups.shape()!=[i32::try_from(source.chunks.rows).map_err(|_|invalid())?,routes]
                || !matches!(groups.dtype(),WorkspaceDtype::Int32|WorkspaceDtype::Uint32){return Err(invalid());}
            let groups=WorkspaceTensor::existing(context.layout(groups.shape(),groups.dtype())?
                .with_representation(groups.representation()),&context)?;
            observer.bind_source_groups(&groups)?;Ok::<_,Error>(observer)
        }).transpose()?;
        let (output, bias) = bank
            .trace_with_parameters(
                &borrowed,
                &values[0],
                &selection,
                source.tensor_partitions,
                &context,
                observed.as_mut().map(|v|v as &mut dyn eredu_nn::GroupedUnitObserver<WorkspaceTensor>),
            )?
            .into_parts();
        let mut outputs = context.metadata_vec(1 + usize::from(bias.is_some()))?;
        outputs.push(output);
        outputs.extend(bias);
        let mut output_layouts = context.metadata_vec(outputs.len())?;
        for value in &outputs {
            output_layouts.push(
                context
                    .layout(value.shape(), value.layout().dtype())?
                    .with_representation(value.layout().representation()),
            );
        }
        let unit_representation=observed.as_ref().and_then(|v|v.representation());
        let (report,capture)=match observed {
            Some(observed)=>observed.finish_report(&outputs)?,
            None=>(context.finish_report(&outputs)?,crate::backend::array_copy::CaptureNativePopulation::default()),
        };
        Ok(Self {
            report,
            output_layouts,capture,unit_representation,
        })
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
