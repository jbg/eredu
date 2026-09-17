use super::*;
const COMPONENT_ABSENT: &str = "pooling component is not declared";
const WIDTH_ABSENT: &str = "pooling component lacks fixed width";
const INVALID_GEOMETRY: &str = "pooling tensor differs from declared batch, width or dtype";
pub(super) fn validation_error_bytes() -> Option<usize> {
    [COMPONENT_ABSENT, WIDTH_ABSENT, INVALID_GEOMETRY]
        .into_iter()
        .map(|message| WorkspaceContext::metadata_error_bytes(message.len()))
        .try_fold(0usize, |maximum, bytes| Some(maximum.max(bytes?)))
}

#[derive(Debug, Clone)]
pub(super) struct WorkspacePoolingStream {
    pub(super) ratio: i32,
    pub(super) position: i32,
    pub(super) batch: i32,
    pub(super) policy: Rc<LayerCachePolicy>,
    pub(super) declarations: [Option<usize>; 5],
    pub(super) values: [Option<WorkspaceTensor>; 5],
}

impl WorkspacePoolingStream {
    pub(super) fn validate(
        &self,
        slot: usize,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        context.validate_values([value])?;
        let declaration = self.declarations[slot]
            .map(|index| &self.policy.fixed_state()[index])
            .ok_or_else(|| context.metadata_error(format_args!("{COMPONENT_ABSENT}")))?;
        let width = match declaration.shape.last() {
            Some(StateTensorDimension::Fixed(width)) => {
                i32::try_from(width.get()).map_err(|cause| context.metadata_source(cause))?
            }
            _ => {
                return Err(context.metadata_error(format_args!("{WIDTH_ABSENT}")));
            }
        };
        let dtype = match declaration.dtype {
            StateTensorDtype::Floating | StateTensorDtype::Float32 => WorkspaceDtype::Float32,
            StateTensorDtype::Int32 => WorkspaceDtype::Int32,
            StateTensorDtype::Uint32 => WorkspaceDtype::Uint32,
        };
        if value.shape().len() != 3
            || value.shape()[0] != self.batch
            || value.shape()[2] != width
            || value.layout().dtype() != dtype
        {
            return Err(context.metadata_error(format_args!("{INVALID_GEOMETRY}")));
        }
        Ok(())
    }

    pub(super) fn accumulate(
        &mut self,
        values: WorkspaceTensor,
        gates: WorkspaceTensor,
        offset: i32,
        context: &WorkspaceContext,
    ) -> Result<PoolingWindows<WorkspaceTensor>, Error> {
        self.validate(0, &values, context)?;
        self.validate(1, &gates, context)?;
        if offset != self.position || values.shape()[1] != gates.shape()[1] {
            return Err(context.metadata_error(format_args!(
                "pooling values/gates or source frontier disagree"
            )));
        }
        let next = self
            .position
            .checked_add(values.shape()[1])
            .ok_or_else(|| context.metadata_error(format_args!("pooling position overflow")))?;
        let previous = self.values[0].as_ref().map_or(0, |value| value.shape()[1]);
        let append = |old: &Option<WorkspaceTensor>, new| match old {
            Some(old) => WorkspaceTensor::concatenate(&[old.clone(), new], 1, context),
            None => Ok(new),
        };
        let values = append(&self.values[0], values)?;
        let gates = append(&self.values[1], gates)?;
        let total = values.shape()[1];
        let usable = total / self.ratio * self.ratio;
        let slice = |value: &WorkspaceTensor, start, end| {
            value.index(
                &[Index::Full, Index::Range(start, end), Index::Full],
                context,
            )
        };
        let ready_values = slice(&values, 0, usable)?;
        let ready_gates = slice(&gates, 0, usable)?;
        let pending_values = (usable < total)
            .then(|| slice(&values, usable, total))
            .transpose()?;
        let pending_gates = (usable < total)
            .then(|| slice(&gates, usable, total))
            .transpose()?;
        self.position = next;
        self.values[0] = pending_values;
        self.values[1] = pending_gates;
        Ok(PoolingWindows {
            values: ready_values,
            gates: ready_gates,
            base_position: offset - previous,
        })
    }

    pub(super) fn append(
        &mut self,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.validate(2, &values, context)?;
        let old_count = self.values[2].as_ref().map_or(0, |value| value.shape()[1]);
        if old_count.checked_add(values.shape()[1]) != Some(self.position / self.ratio) {
            return Err(context.metadata_error(format_args!(
                "pooled history does not match completed source windows"
            )));
        }
        if values.shape()[1] > 0 {
            self.values[2] = Some(match &self.values[2] {
                Some(old) => WorkspaceTensor::concatenate(&[old.clone(), values], 1, context)?,
                None => values,
            });
        } else if self.values[2].is_none() {
            return values.zeros_like(context);
        }
        Ok(self.values[2]
            .as_ref()
            .expect("retained pooled history")
            .clone())
    }
}
