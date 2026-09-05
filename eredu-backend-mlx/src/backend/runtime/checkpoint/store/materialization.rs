use super::*;

/// Scheduled tensor materialization that still pins its mmap-backed sources.
pub struct PendingWeightMaterialization {
    pub(super) output: Array,
    pub(super) _source: Array,
    pub(super) _gguf_group: Option<Arc<CachedGgufGroup>>,
    pub(super) lease: Option<WeightLease>,
    pub(super) source_stream: Stream,
    pub(super) execution_stream: Stream,
    pub(super) borrowed_source: bool,
    pub(super) completed: bool,
}

impl PendingWeightMaterialization {
    /// Returns the lazy materialized output.
    pub fn output(&self) -> &Array {
        &self.output
    }

    #[cfg(test)]
    pub(super) fn finish(self) -> Result<Array, CheckpointMaterializationError> {
        self.submit()?.synchronize()
    }

    pub(super) fn submit(
        mut self,
    ) -> Result<WeightMaterialization, CheckpointMaterializationError> {
        self.prepare_owned_output()?;
        let output = self.output.clone();
        WeightMaterialization::submit_retained(output, vec![self])
    }

    fn prepare_owned_output(&mut self) -> Result<(), CheckpointMaterializationError> {
        if !self.borrowed_source {
            return Ok(());
        }
        let output = self.output.copy(&self.source_stream).map_err(|source| {
            self.lease
                .as_ref()
                .expect("pending materialization retains its lease")
                .mlx_error("borrowed source copy", source)
        })?;
        self.output = output;
        self.borrowed_source = false;
        Ok(())
    }

    fn complete_in_place(&mut self) {
        self.completed = true;
        self.lease.take();
    }

    /// Marks a batch member complete after a containing output was evaluated.
    pub fn complete(mut self) {
        self.completed = true;
        self.lease.take();
    }
}

/// Owning completion for one checkpoint tensor materialization.
///
/// This single-shot guard retains checkpoint leases and source arrays until
/// its exact MLX completion finishes. It may order multiple compatible
/// consumers. Dropping an unfinished guard blocks only for this event, never
/// for an entire stream. Asynchronous backend errors are returned by query or
/// synchronization. The type is intentionally neither `Send` nor `Sync`
/// because it owns `safemlx`'s thread-affine [`Event`].
#[must_use = "checkpoint leases remain retained until this completion is consumed or dropped"]
pub struct WeightMaterialization {
    output: Array,
    sources: Vec<PendingWeightMaterialization>,
    event: Option<Event>,
}

impl WeightMaterialization {
    /// Submits an output and retains its source materializations until completion.
    pub fn submit_retained(
        output: Array,
        sources: Vec<PendingWeightMaterialization>,
    ) -> Result<Self, CheckpointMaterializationError> {
        let key = sources
            .first()
            .and_then(|pending| pending.lease.as_ref())
            .map(|lease| lease.key().to_owned())
            .unwrap_or_else(|| "<derived checkpoint materialization>".into());
        let event = async_eval_with_event([&output]).map_err(|source| {
            CheckpointMaterializationError::Mlx {
                key,
                operation: "evaluation submission",
                source,
            }
        })?;
        Ok(Self {
            output,
            sources,
            event: Some(event),
        })
    }

    /// Returns the materialized output while this guard retains its sources.
    pub fn output(&self) -> &Array {
        &self.output
    }

    /// Orders subsequently submitted work on `stream` after this completion.
    ///
    /// This does not block the host. The stream must be backend/device
    /// compatible, and the consumer graph must be evaluated after this call.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), CheckpointMaterializationError> {
        self.event
            .as_ref()
            .expect("unfinished materialization retains its event")
            .wait_on(stream)
            .map_err(|source| self.mlx_error("consumer stream wait", source))
    }

    /// Returns whether the exact materialization has completed without blocking.
    pub fn is_complete(&self) -> Result<bool, CheckpointMaterializationError> {
        self.event
            .as_ref()
            .expect("unfinished materialization retains its event")
            .is_complete()
            .map_err(|source| self.mlx_error("completion query", source))
    }

    /// Blocks for the exact completion and returns the independently owned output.
    pub fn synchronize(mut self) -> Result<Array, CheckpointMaterializationError> {
        let output = self.output().clone();
        self.finish(true)?;
        Ok(output)
    }

    fn finish(&mut self, report_error: bool) -> Result<(), CheckpointMaterializationError> {
        let Some(event) = self.event.take() else {
            return Ok(());
        };
        let key = self
            .sources
            .first()
            .and_then(|pending| pending.lease.as_ref())
            .map(|lease| lease.key().to_owned())
            .unwrap_or_else(|| "<completed checkpoint materialization>".into());
        let result = event.synchronize();
        for mut pending in self.sources.drain(..) {
            pending.complete_in_place();
        }
        match result {
            Ok(()) => Ok(()),
            Err(source) if report_error => Err(CheckpointMaterializationError::Mlx {
                key,
                operation: "completion",
                source,
            }),
            Err(_) => Ok(()),
        }
    }

    fn mlx_error(
        &self,
        operation: &'static str,
        source: safemlx::error::Exception,
    ) -> CheckpointMaterializationError {
        let key = self
            .sources
            .first()
            .and_then(|pending| pending.lease.as_ref())
            .map(|lease| lease.key().to_owned())
            .unwrap_or_else(|| "<completed checkpoint materialization>".into());
        CheckpointMaterializationError::Mlx {
            key,
            operation,
            source,
        }
    }
}

impl Drop for WeightMaterialization {
    fn drop(&mut self) {
        let _ = self.finish(false);
    }
}

impl Drop for PendingWeightMaterialization {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        // Submission creates the exact completion event and moves this value's
        // checkpoint lease into `WeightMaterialization`. If submission itself is
        // abandoned or fails before that event exists, draining both candidate
        // streams is the only conservative way to prove that no lazy copy still
        // references the mmap. This error-cleanup path is intentionally the sole
        // whole-stream wait in the eredu runtime.
        let source = self.source_stream.synchronize();
        let execution = self.execution_stream.synchronize();
        if source.is_err() || execution.is_err() {
            if let Some(lease) = &self.lease {
                lease.retain_mapping_after_sync_failure();
            }
        }
    }
}

pub(super) fn materialize_contiguous(
    key: &str,
    source: &Array,
    offset_elements: usize,
    shape: &[usize],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, CheckpointMaterializationError> {
    let elements = shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| {
            CheckpointMaterializationError::ArithmeticOverflow {
                context: format!("contiguous materialization size for tensor {key:?}"),
            }
        })
    })?;
    let end = offset_elements.checked_add(elements).ok_or_else(|| {
        CheckpointMaterializationError::ArithmeticOverflow {
            context: format!("contiguous materialization end for tensor {key:?}"),
        }
    })?;
    let flattened = source.reshape(&[-1], source_stream).map_err(|source| {
        CheckpointMaterializationError::Mlx {
            key: key.to_string(),
            operation: "flatten contiguous selection",
            source,
        }
    })?;
    let selected = materialize_range(
        key,
        flattened,
        &[source.size()],
        0,
        offset_elements,
        end,
        source_stream,
        execution_stream,
    )?;
    let shape = shape
        .iter()
        .map(|dimension| to_i32(key, "contiguous output dimension", *dimension))
        .collect::<Result<Vec<_>, _>>()?;
    selected
        .reshape(&shape, execution_stream)
        .map_err(|source| CheckpointMaterializationError::Mlx {
            key: key.to_string(),
            operation: "reshape contiguous selection",
            source,
        })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_range(
    key: &str,
    source: Array,
    source_shape: &[usize],
    axis: usize,
    start: usize,
    end: usize,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, CheckpointMaterializationError> {
    let axis_i32 = to_i32(key, "axis", axis)?;
    let front = if axis == 0 {
        source
    } else {
        source
            .move_axis(axis_i32, 0, source_stream)
            .map_err(|source| mlx_error(key, "move range axis", source))?
    };
    let start = to_i32(key, "range start", start)?;
    let end = to_i32(key, "range end", end)?;
    let selected = front
        .try_index_device(start..end, source_stream)
        .map_err(|source| mlx_error(key, "range selection", source))?;
    let selected = if axis == 0 {
        selected
    } else {
        selected
            .move_axis(0, axis_i32, source_stream)
            .map_err(|source| mlx_error(key, "restore range axis", source))?
    };
    let selected = if axis == 0 {
        selected
    } else {
        // Inner-axis ranges are non-contiguous views. Compact only the selected
        // result, keeping the temporary bounded by the output shape.
        let mut output_shape = source_shape.to_vec();
        output_shape[axis] = usize::try_from(end - start).map_err(|_| {
            CheckpointMaterializationError::ArithmeticOverflow {
                context: format!("selected range length for tensor {key:?}"),
            }
        })?;
        let row_major_shape = output_shape
            .iter()
            .map(|dimension| to_i32(key, "selected dimension", *dimension))
            .collect::<Result<Vec<_>, _>>()?;
        selected
            .flatten(None, None, source_stream)
            .and_then(|value| value.reshape(&row_major_shape, source_stream))
            .map_err(|source| mlx_error(key, "range compaction", source))?
    };
    selected
        .copy(execution_stream)
        .map_err(|source| mlx_error(key, "copy", source))
}

pub(super) fn materialize_indices(
    key: &str,
    source: &Array,
    axis: usize,
    indices: &[usize],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, CheckpointMaterializationError> {
    let axis = to_i32(key, "axis", axis)?;
    let indices = indices
        .iter()
        .map(|index| to_i32(key, "tensor index", *index))
        .collect::<Result<Vec<_>, _>>()?;
    let count = to_i32(key, "index count", indices.len())?;
    let index_array = Array::from_slice(&indices, &[count])
        .copy(source_stream)
        .map_err(|source| mlx_error(key, "index upload", source))?;
    source
        .take_axis(&index_array, axis, source_stream)
        .and_then(|selected| selected.copy(execution_stream))
        .map_err(|source| mlx_error(key, "ordered index selection", source))
}

fn to_i32(
    key: &str,
    what: &'static str,
    value: usize,
) -> Result<i32, CheckpointMaterializationError> {
    i32::try_from(value).map_err(|_| CheckpointMaterializationError::ArithmeticOverflow {
        context: format!("{what} for tensor {key:?} does not fit in i32"),
    })
}

fn mlx_error(
    key: &str,
    operation: &'static str,
    source: safemlx::error::Exception,
) -> CheckpointMaterializationError {
    CheckpointMaterializationError::Mlx {
        key: key.to_string(),
        operation,
        source,
    }
}
