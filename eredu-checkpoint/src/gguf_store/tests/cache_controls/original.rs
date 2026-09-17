//! Unchanged complete pre-G4 cache body; only the method name differs.
use super::*;
impl ReferenceCache {
    pub(super) fn old_g4_materialize_with_destination(
        &mut self,
        checkpoint: usize,
        physical_name: &str,
        selection: Option<&GgufPhysicalSelection>,
        maximum: usize,
        logical_key: &str,
        raw: Option<&mut [u8]>,
        conversion: Option<&mut eredu_gguf::PreparedConversion>,
        metadata: Option<&mut eredu_gguf::PreparedTensorMetadata>,
    ) -> Result<ConvertedCheckpointTensor, read_destination::Cause> {
        let target_path = self
            .materializers
            .get(checkpoint)
            .ok_or_else(|| gguf_error(logical_key, "catalog references an unknown checkpoint"))?
            .shard_path_for_tensor(physical_name)
            .map_err(|error| gguf_error(logical_key, error))?
            .to_path_buf();
        let reader_hit = self.materializers[checkpoint]
            .open_shard_path()
            .is_some_and(|path| path == target_path);
        self.tick = self.tick.saturating_add(1);
        if reader_hit {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
            if self.materializers[checkpoint].close_reader().is_some() {
                self.evictions = self.evictions.saturating_add(1);
            }
            if self
                .materializers
                .iter()
                .filter(|materializer| materializer.open_shard_path().is_some())
                .count()
                >= maximum
            {
                let victim = self
                    .materializers
                    .iter()
                    .enumerate()
                    .filter(|(_, materializer)| materializer.open_shard_path().is_some())
                    .min_by_key(|(index, _)| (self.last_used[*index], *index))
                    .map(|(index, _)| index)
                    .expect("an open reader exists at the configured bound");
                self.materializers[victim].close_reader();
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        self.last_used[checkpoint] = self.tick;
        let materializer = &mut self.materializers[checkpoint];
        let converted = if let Some(metadata) = metadata {
            let raw = raw.expect("complete metadata is paired with its raw owner");
            let conversion =
                conversion.expect("complete metadata is paired with conversion storage");
            let selection = match selection {
                None => eredu_gguf::MetadataSelection::Full,
                Some(GgufPhysicalSelection::Axis(s)) => eredu_gguf::MetadataSelection::Axis(s),
                Some(GgufPhysicalSelection::DenseSpan(s)) => eredu_gguf::MetadataSelection::Span(s),
            };
            materializer.converted_tensor_with_metadata(
                physical_name,
                selection,
                raw,
                conversion,
                metadata,
            )
        } else if let Some(conversion) = conversion {
            let raw = raw.expect("prepared conversion is paired with its raw owner");
            match selection {
                Some(GgufPhysicalSelection::Axis(selection)) => materializer
                    .converted_tensor_selected_with_destinations(
                        physical_name,
                        selection,
                        raw,
                        conversion,
                    ),
                Some(GgufPhysicalSelection::DenseSpan(selection)) => materializer
                    .converted_dense_tensor_span_with_destinations(
                        physical_name,
                        selection,
                        raw,
                        conversion,
                    ),
                None => {
                    materializer.converted_tensor_with_destinations(physical_name, raw, conversion)
                }
            }
        } else {
            match (selection, raw) {
                (Some(GgufPhysicalSelection::Axis(selection)), None) => materializer
                    .converted_tensor_selected(physical_name, selection)
                    .map_err(eredu_gguf::ReadDestinationError::from),
                (Some(GgufPhysicalSelection::DenseSpan(selection)), None) => materializer
                    .converted_dense_tensor_span(physical_name, selection)
                    .map_err(eredu_gguf::ReadDestinationError::from),
                (None, None) => materializer
                    .converted_tensor(physical_name)
                    .map_err(eredu_gguf::ReadDestinationError::from),
                (Some(GgufPhysicalSelection::Axis(selection)), Some(raw)) => materializer
                    .converted_tensor_selected_with_raw_destination(physical_name, selection, raw),
                (Some(GgufPhysicalSelection::DenseSpan(selection)), Some(raw)) => materializer
                    .converted_dense_tensor_span_with_raw_destination(
                        physical_name,
                        selection,
                        raw,
                    ),
                (None, Some(raw)) => {
                    materializer.converted_tensor_with_raw_destination(physical_name, raw)
                }
            }
        }
        .map_err(|error| read_destination::Cause::read(error, logical_key))?;
        self.touched.insert(target_path);
        Ok(converted)
    }
}
