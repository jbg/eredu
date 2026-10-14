//! Complete Pixtral host-buffer admission before floating cubic preprocessing.
use super::*;
use crate::nemotron_labs_diffusion::processing::ImageProcessor;

pub(super) struct Request<'a> {
    plan: &'a ImageProcessor,
    request: &'a TokenizedMultimodalRequest,
    pub resources: ProcessorRequestResources,
}

impl<'a> Request<'a> {
    pub fn plan<E: std::fmt::Display>(
        plan: &'a ImageProcessor,
        request: &'a TokenizedMultimodalRequest,
        budget: Option<ProcessorRequestBudget>,
    ) -> Result<Self, ProcessorExecutionError<E, eredu_media::MediaError>> {
        let overflow = || ProcessorResourceError::Overflow;
        let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(overflow);
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(overflow);
        let mut resources = ProcessorRequestResources::default();
        let mut scratch = 0;
        resources.planning_items = mul(request.segments().len() as u64, 24)?;
        for segment in request.segments() {
            let (decoded, output, positions, temporary) = match segment {
                TokenizedMultimodalSegment::TokenIds(ids) => {
                    let bytes = mul(ids.len() as u64, 4)?;
                    (bytes, bytes, ids.len() as u64, bytes)
                }
                TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                    let (height, width) = plan
                        .grid(image.width(), image.height())
                        .map_err(|error| ProcessorExecutionError::Plan(error.to_string()))?;
                    let patches = mul(height.into(), width.into())?;
                    let floats = mul(mul(patches, plan.patch_vector_width() as u64)?, 4)?;
                    let columns = mul(width.into(), plan.patch_size().into())?;
                    // Normalized pixels, packed patches, host constructor copy;
                    // cubic column weights and at most four interpolated rows.
                    let temporary = add(mul(floats, 3)?, mul(columns, 160)?)?;
                    let merge = u64::from(plan.merge_size());
                    let positions = add(
                        1,
                        mul(u64::from(height) / merge, add(u64::from(width) / merge, 1)?)?,
                    )?;
                    (
                        image.pixels().len() as u64,
                        add(floats, 12)?,
                        positions,
                        add(temporary, mul(positions, 8)?)?,
                    )
                }
                TokenizedMultimodalSegment::Media(_) => {
                    return Err(ProcessorExecutionError::Plan(
                        "Pixtral image processing accepts text and still images".into(),
                    ));
                }
            };
            resources.decoded_input_bytes = add(resources.decoded_input_bytes, decoded)?;
            resources.output_tensor_bytes = add(resources.output_tensor_bytes, output)?;
            resources.decoder_positions = add(resources.decoder_positions, positions)?;
            resources.planning_items = add(resources.planning_items, positions)?;
            scratch = scratch.max(temporary);
            resources.host_buffer_bytes = add(resources.output_tensor_bytes, scratch)?;
            if let Some(budget) = budget {
                budget.admit(&resources)?;
            }
        }
        Ok(Self {
            plan,
            request,
            resources,
        })
    }

    pub fn execute<E: std::fmt::Display>(
        self,
        mechanisms: &mut HostProcessorMechanisms,
    ) -> Result<PreparedModelInput<HostTensor>, ProcessorExecutionError<E, eredu_media::MediaError>>
    {
        let mut parts = Vec::with_capacity(self.request.segments().len());
        for segment in self.request.segments() {
            match segment {
                TokenizedMultimodalSegment::TokenIds(ids) => {
                    push_tokens(&mut parts, ids, mechanisms)?
                }
                TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                    let image = self
                        .plan
                        .prepare(
                            eredu_media::image::RgbImageView::packed(
                                image.pixels(),
                                image.width(),
                                image.height(),
                            )
                            .map_err(ProcessorExecutionError::Mechanism)?,
                        )
                        .map_err(|error| ProcessorExecutionError::Plan(error.to_string()))?;
                    let (height, width) = image.grid;
                    let tensor = mechanisms
                        .tensor_f32(
                            &image.patches,
                            &[
                                height as usize * width as usize,
                                self.plan.patch_vector_width(),
                            ],
                        )
                        .map_err(mechanism_error)?;
                    let grid = mechanisms
                        .tensor_i32(&[1, height as i32, width as i32], &[1, 3])
                        .map_err(mechanism_error)?;
                    parts.push(
                        PreparedInputPart::new_with_extents(
                            InputModality::Image,
                            PreparedInputPayload::Tensor(tensor),
                            [(InputMetadataKey::PatchGrid, grid)],
                            [InputExtent::PatchGrid {
                                time: 1,
                                height: height as usize,
                                width: width as usize,
                            }],
                        )
                        .map_err(ProcessorExecutionError::Prepared)?,
                    );
                }
                TokenizedMultimodalSegment::Media(_) => {
                    unreachable!("request modalities were admitted")
                }
            }
        }
        PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
            .map_err(ProcessorExecutionError::Prepared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{MultimodalRequest, MultimodalSegment};

    #[test]
    fn published_pixels_lower_without_duplicate_markers_and_admit_buffers_before_processing() {
        let fixture: serde_json::Value = serde_json::from_str(
            eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
        )
        .unwrap();
        let config =
            crate::nemotron_labs_diffusion::FamilyConfig::from_hf_value(&fixture["config"])
                .unwrap();
        let processor = ImageProcessor::new(&config, 16).unwrap();
        let reference = &fixture["processor"][0];
        let pixels = reference["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let request = MultimodalRequest::new(vec![
            MultimodalSegment::TokenIds(vec![1, 2]),
            MultimodalSegment::Media(Media::Image(RgbImage::new(pixels, 3, 5).unwrap())),
            MultimodalSegment::TokenIds(vec![4]),
        ])
        .unwrap()
        .tokenize::<std::convert::Infallible>(|_| unreachable!())
        .unwrap();
        let prepared =
            Request::plan::<std::convert::Infallible>(&processor, &request, None).unwrap();
        let resources = prepared.resources;
        assert_eq!(resources.decoder_positions, 8);
        let budget = ProcessorRequestBudget {
            decoded_input_bytes: resources.decoded_input_bytes,
            host_buffer_bytes: resources.host_buffer_bytes - 1,
            output_tensor_bytes: resources.output_tensor_bytes,
            planning_items: resources.planning_items,
            decoder_positions: resources.decoder_positions,
        };
        assert!(matches!(
            Request::plan::<std::convert::Infallible>(&processor, &request, Some(budget)),
            Err(ProcessorExecutionError::Resources(
                ProcessorResourceError::Exhausted {
                    resource: "host buffer bytes",
                    ..
                }
            ))
        ));
        let input = prepared
            .execute::<std::convert::Infallible>(&mut HostProcessorMechanisms)
            .unwrap();
        assert_eq!(
            input.parts().len(),
            3,
            "the architecture inserts markers around each image part"
        );
        let HostTensor::F32 { values, shape } = input.parts()[1].payload().value() else {
            unreachable!()
        };
        assert_eq!(shape, &[8, 12]);
        for (actual, expected) in values
            .iter()
            .zip(reference["patches"]["data"].as_array().unwrap())
        {
            assert!((*actual - expected.as_f64().unwrap() as f32).abs() < 0.000003);
        }
        let admitted = crate::nemotron_labs_diffusion::input::admit_input(
            &config,
            &input,
            &HostProcessorMechanisms,
        )
        .unwrap();
        assert_eq!(admitted.decoder_positions(), resources.decoder_positions);
    }
}
