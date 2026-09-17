//! Existing native cache storage projected into the selected portable mechanism.

use super::CompressedLatentCache;
use crate::backend::nn::workspace::ExistingArrayProjection;
use eredu_nn::{CompressedAttentionState, Error, workspace::WorkspaceContext};
use eredu_runtime::working_memory::WorkspaceCompressedCache;
use std::num::NonZeroU32;

impl CompressedLatentCache {
    /// Projects actual resident storage without evaluation, native allocation or
    /// prefix replay. The declared local geometry comes from architecture state
    /// selection; this adapter supplies only native capacity and alias facts.
    /// Lazy or foreign-owned backing remains unknown in the resulting quote.
    /// Paged storage requires its separate manager/block inventory.
    pub fn project_resident_workspace(
        &self,
        batch: NonZeroU32,
        latent_width: NonZeroU32,
        rotary_width: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceCompressedCache, Error> {
        self.project_workspace_state(
            batch,
            latent_width,
            rotary_width,
            &mut ExistingArrayProjection::new(context),
        )
    }

    pub(crate) fn project_workspace_state<'a>(
        &'a self,
        batch: NonZeroU32,
        latent_width: NonZeroU32,
        rotary_width: NonZeroU32,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<WorkspaceCompressedCache, Error> {
        self.project_workspace_state_with(
            batch,
            latent_width,
            rotary_width,
            projection.context(),
            |array| projection.project(array),
        )
    }

    pub(crate) fn project_workspace_state_with<'a, F>(
        &'a self,
        batch: NonZeroU32,
        latent_width: NonZeroU32,
        rotary_width: NonZeroU32,
        context: &WorkspaceContext,
        mut project: F,
    ) -> Result<WorkspaceCompressedCache, Error>
    where
        F: FnMut(&'a safemlx::Array) -> Result<eredu_nn::workspace::WorkspaceTensor, Error>,
    {
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            NonZeroU32,
            NonZeroU32,
            NonZeroU32,
            &WorkspaceContext,
            F,
            WorkspaceCompressedCache,
            Result<WorkspaceCompressedCache, Error>,
        )>())?;
        let step = self.workspace_source_step(context)?;
        let projected =
            WorkspaceCompressedCache::new(batch, latent_width, rotary_width, step, context)?;
        match (
            &self.latent_storage,
            &self.rotary_key_storage,
            &self.latent,
            &self.rotary_key,
        ) {
            (None, None, None, None) if self.length == 0 && self.capacity == 0 => Ok(projected),
            (Some(latent_store), Some(rotary_store), Some(latent), Some(rotary)) => {
                let projected = projected.with_existing_state(
                    CompressedAttentionState {
                        latent: project(latent_store)?,
                        rotary: project(rotary_store)?,
                    },
                    CompressedAttentionState {
                        latent: project(latent)?,
                        rotary: project(rotary)?,
                    },
                )?;
                use eredu_nn::CompressedAttentionCache;
                if projected.offset() != self.offset || projected.capacity() != self.capacity {
                    return Err(context.metadata_error(format_args!(
                        "native compressed shapes disagree with retained frontier/capacity"
                    )));
                }
                Ok(projected)
            }
            _ => Err(context
                .metadata_error(format_args!("incomplete native compressed state inventory"))),
        }
    }
    fn workspace_source_step(&self, context: &WorkspaceContext) -> Result<NonZeroU32, Error> {
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &WorkspaceContext,
            Result<NonZeroU32, Error>,
        )>())?;
        if self.paged.is_some() {
            return Err(context.metadata_error(format_args!(
                "paged compressed workspace requires a complete manager and block projection"
            )));
        }
        if self.offset != self.length || self.length < 0 || self.capacity < self.length {
            return Err(
                context.metadata_error(format_args!("invalid resident compressed state frontier"))
            );
        }
        u32::try_from(self.step)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| {
                context.metadata_error(format_args!("invalid native compressed capacity step"))
            })
    }
    pub(crate) fn workspace_source_count(
        &self,
        context: &WorkspaceContext,
    ) -> Result<usize, Error> {
        self.workspace_source_step(context)?;
        let values = self.borrowed_retained_values();
        context.charge_metadata(
            std::mem::size_of_val(&values)
                .checked_add(std::mem::size_of::<(
                    &Self,
                    &WorkspaceContext,
                    usize,
                    Result<usize, Error>,
                )>())
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        Ok(values.count())
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
    use eredu_nn::{CompressedAttentionCache, Tensor, workspace::WorkspaceTensor};
    use safemlx::{Array, Device, DeviceType, Stream};
    use std::collections::BTreeMap;

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }
    fn arrays(cache: &CompressedLatentCache) -> Vec<&Array> {
        cache
            .latent_storage
            .iter()
            .chain(&cache.rotary_key_storage)
            .chain(&cache.latent)
            .chain(&cache.rotary_key)
            .collect()
    }
    fn complete(cache: &CompressedLatentCache) {
        for array in arrays(cache) {
            array.evaluated().unwrap();
        }
    }
    fn payload(start: i32, count: i32, width: i32) -> Vec<f32> {
        (0..2)
            .flat_map(|batch| {
                (start..start + count).flat_map(move |position| {
                    (0..width)
                        .map(move |channel| (batch * 1000 + position * 10 + channel + 1) as f32)
                })
            })
            .collect()
    }

    #[test]
    #[ignore = "requires MLX Metal execution"]
    fn compressed_existing_projection_prices_restored_capacity_and_continues_without_replay() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let mut source = CompressedLatentCache::new();
        source
            .update_and_fetch(
                Array::from_slice(&payload(0, 3, 8), &[2, 3, 8]),
                Array::from_slice(&payload(0, 3, 4), &[2, 3, 4]),
                &stream,
            )
            .unwrap();
        complete(&source);
        for (kind, mut native) in [
            ("live", source.clone()),
            ("restored", source.deep_clone_state().unwrap()),
            ("compact", source.isolated_snapshot(&stream).unwrap()),
        ] {
            complete(&native);
            let allocations: BTreeMap<_, _> = arrays(&native)
                .into_iter()
                .map(|array| {
                    let info = array
                        .allocation_info()
                        .unwrap()
                        .expect("completed owned native backing");
                    (info.identity(), info.bytes() as u64)
                })
                .collect();
            assert_eq!(allocations.len(), if kind == "restored" { 4 } else { 2 });
            let existing_bytes = allocations.values().sum::<u64>();
            let detached_view_bytes = if kind == "restored" {
                let (latent, rotary) = native.arrays().unwrap();
                [latent, rotary]
                    .into_iter()
                    .map(|array| array.allocation_info().unwrap().unwrap().bytes() as u64)
                    .sum()
            } else {
                0
            };
            let context =
                WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
            let mut projection = ExistingArrayProjection::new(&context);
            let mut metadata = native
                .project_workspace_state(nz(2), nz(8), nz(4), &mut projection)
                .unwrap();
            let storage = projection.into_storage();
            assert!(storage.is_complete());
            assert_eq!(
                storage
                    .iter()
                    .map(|(identity, bytes, root)| {
                        assert_eq!(root.capacity_bytes(), Some(bytes));
                        (identity, bytes)
                    })
                    .collect::<BTreeMap<_, _>>(),
                allocations
            );
            assert_eq!(
                (metadata.offset(), metadata.capacity()),
                (native.offset(), native.capacity())
            );
            context
                .begin_state_span(metadata.retained_arrays())
                .unwrap();
            let roots = metadata.retained_arrays().cloned().collect::<Vec<_>>();
            let initial = context.report(&roots).unwrap();
            assert!(initial.operations.is_empty());
            assert_eq!(initial.state.unwrap().retained_bytes, Some(existing_bytes));
            drop(roots);
            // Binding witnesses must not pin retired native state during the
            // subsequent capacity transitions exercised below.
            drop(storage);
            for count in [1, 252, 2] {
                let start = native.offset();
                context
                    .begin_state_span(metadata.retained_arrays())
                    .unwrap();
                let latent = payload(start, count, 8);
                let rotary = payload(start, count, 4);
                metadata
                    .append(
                        CompressedAttentionState {
                            latent: WorkspaceTensor::from_f32_slice(
                                &latent,
                                &[2, count, 8],
                                &context,
                            )
                            .unwrap(),
                            rotary: WorkspaceTensor::from_f32_slice(
                                &rotary,
                                &[2, count, 4],
                                &context,
                            )
                            .unwrap(),
                        },
                        &context,
                    )
                    .unwrap();
                native
                    .update_and_fetch(
                        Array::from_slice(&latent, &[2, count, 8]),
                        Array::from_slice(&rotary, &[2, count, 4]),
                        &stream,
                    )
                    .unwrap();
                complete(&native);
                assert_eq!(
                    (metadata.offset(), metadata.capacity()),
                    (native.offset(), native.capacity())
                );
                let report = context
                    .report(&metadata.retained_arrays().cloned().collect::<Vec<_>>())
                    .unwrap();
                assert!(report.inference_transient_bytes().is_some());
                if start == 3 {
                    let state = report.state.as_ref().unwrap();
                    assert_eq!(state.displaced_bytes, Some(detached_view_bytes));
                    assert!(state.retained_bytes.unwrap() + detached_view_bytes >= existing_bytes);
                }
                let (latent, rotary) = native.arrays().unwrap();
                for (array, width) in [(latent, 8), (rotary, 4)] {
                    let packed = array.contiguous(false, &stream).unwrap();
                    assert_eq!(
                        packed.evaluated().unwrap().as_slice::<f32>(),
                        payload(0, native.offset(), width)
                    );
                }
            }
        }
    }
}
