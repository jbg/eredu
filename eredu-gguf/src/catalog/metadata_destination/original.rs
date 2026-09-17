//! Pre-G3 materializer bodies; only helper names and the closed reader-view adapter differ.
use super::*;
impl TensorMaterializer {
    pub(super) fn old_location_and_reader(
        &mut self,
        name: &str,
    ) -> Result<(TensorLocation, TensorDescriptor, Endian)> {
        let location =
            self.locations
                .get(name, &self.checkpoint)
                .ok_or_else(|| Error::InvalidTensor {
                    tensor: name.to_string(),
                    reason: "tensor is not present in the checkpoint".into(),
                })?;
        let shard = &self.checkpoint.shards[location.shard_index];
        if self
            .reader
            .as_ref()
            .is_none_or(|(shard_index, _)| *shard_index != location.shard_index)
        {
            let reader = validate_reopened_shard(
                open_reader(&shard.path, self.checkpoint.limits.clone())?,
                shard,
            )?;
            self.reader = Some((location.shard_index, reader));
        }
        Ok((
            location,
            shard.tensors[location.tensor_index].descriptor.clone(),
            shard.endian,
        ))
    }
    pub(super) fn old_catalog_output_names(&self, location: TensorLocation) -> Vec<String> {
        self.checkpoint.shards[location.shard_index].tensors[location.tensor_index]
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }
    pub(super) fn old_converted_tensor_with_storage(
        &mut self,
        name: &str,
        storage: RawStorage<'_>,
        conversion: Option<&mut crate::PreparedConversion>,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        let (location, descriptor, _) = self.old_location_and_reader(name)?;
        let output_names = self.old_catalog_output_names(location);
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_tensor_with_storage(&descriptor, storage, conversion)
            .map_err(|source| {
                source.with_shard(&self.checkpoint.shards[location.shard_index].path)
            })?;
        Ok(ConvertedCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            descriptor,
            output_names,
            converted,
        })
    }
    pub(super) fn old_converted_tensor_selected_with_storage(
        &mut self,
        name: &str,
        selection: &TensorSelection,
        storage: RawStorage<'_>,
        conversion: Option<&mut crate::PreparedConversion>,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        let (location, mut descriptor, _) = self.old_location_and_reader(name)?;
        let output_names = self.old_catalog_output_names(location);
        let plan = TensorSelectionPlan::new(&descriptor, selection.clone())?;
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_tensor_plan_with_storage(plan.view(), storage, conversion)
            .map_err(|source| {
                source.with_shard(&self.checkpoint.shards[location.shard_index].path)
            })?;
        descriptor = plan.selected_descriptor().clone();
        Ok(ConvertedCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            descriptor,
            output_names,
            converted,
        })
    }
    pub(super) fn old_converted_dense_tensor_span_with_storage(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
        storage: RawStorage<'_>,
        conversion: Option<&mut crate::PreparedConversion>,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        let (location, descriptor, _) = self.old_location_and_reader(name)?;
        let output_names = self.old_catalog_output_names(location);
        let plan = DenseTensorSpanPlan::new(&descriptor, selection.clone())?;
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_dense_tensor_span_with_storage(plan.view(), storage, conversion)
            .map_err(|source| {
                source.with_shard(&self.checkpoint.shards[location.shard_index].path)
            })?;
        Ok(ConvertedCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            descriptor: plan.selected_descriptor().clone(),
            output_names,
            converted,
        })
    }
}
