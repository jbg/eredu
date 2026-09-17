//! Current prediction state projected through architecture-declared membership.
pub(super) mod capture;
pub(super) mod quote;
pub(super) mod target_quote;
pub(super) mod external_target;
mod target_sources;
pub(super) mod source_bindings;
use super::{MlxEmbeddedPredictionMaterializer, OwnedPredictionCache};
use crate::backend::{
    nn::{
        shared::MlxNeuralBackend,
        workspace::{ExistingArrayProjection, ProjectedNativeStorage},
    },
    runtime::cache::{
        kv::CompressedLatentCache,
        state::{MlxHybridState, MlxPoolingAttentionCache},
    },
};
use eredu_architectures::prediction_extension::{
    MaterializedPredictionExecutor, PredictionStateSourceFactory, PredictionStateSourceLayout,
};
use eredu_core::cache::LayerCachePolicy;
use eredu_nn::{
    Error,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceMetadataError},
};
use eredu_runtime::{
    DeviceState, RuntimeState,
    working_memory::{
        WorkspaceCompressedCache, WorkspacePoolingLayerState, WorkspacePoolingStateFactory,
        WorkspaceResidentLayerState,
    },
};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU32,
};

/// Metadata state of the same current lane. Neither state nor storage authorizes
/// native execution; the consuming invocation still binds source and revision.
pub(crate) enum ProjectedPredictionState {
    Sequential(Vec<WorkspaceCompressedCache>),
    Pooling(Vec<WorkspacePoolingLayerState>),
    Model(DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>),
}

pub(crate) struct ProjectedPredictionLane {
    pub(crate) state: ProjectedPredictionState,
    pub(crate) storage: ProjectedNativeStorage,
}

/// The caller lends the actual typed branch and its selected prepared layout.
/// Source profile dispatch stays in the architecture's existing executor.
pub(crate) fn project_prediction_state<A, P>(
    source: &P::LaneState,
    layout: PredictionStateSourceLayout<'_>,
    batch: NonZeroU32,
    context: &WorkspaceContext,
) -> Result<ProjectedPredictionLane, Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    let frames = [
        size_of::<ProjectionFactory<'_>>(),
        size_of::<ProjectedPredictionLane>(),
        size_of::<Result<ProjectedPredictionLane, Error>>(),
        size_of::<ExistingArrayProjection<'_>>(),
        size_of::<WorkspacePoolingStateFactory>(),
        size_of::<PredictionStateSourceLayout<'_>>(),
    ];
    context.charge_metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    P::prepare_state(
        source,
        &mut ProjectionFactory {
            layout,
            batch,
            context,
        },
    )
}

struct ProjectionFactory<'a> {
    layout: PredictionStateSourceLayout<'a>,
    batch: NonZeroU32,
    context: &'a WorkspaceContext,
}
impl ProjectionFactory<'_> {
    fn mismatch(&self) -> Error {
        self.context.metadata_error(format_args!(
            "current prediction state differs from its prepared source layout"
        ))
    }
}
impl PredictionStateSourceFactory<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
    for ProjectionFactory<'_>
{
    type Error = Error;
    type Prepared<T> = ProjectedPredictionLane;

    fn sequential(
        &mut self,
        source: &[OwnedPredictionCache<CompressedLatentCache>],
    ) -> Result<ProjectedPredictionLane, Error> {
        let PredictionStateSourceLayout::Sequential(policies) = &self.layout else {
            return Err(self.mismatch());
        };
        if policies.len() != source.len() {
            return Err(self.mismatch());
        }
        let mut projection = ExistingArrayProjection::new(self.context);
        let mut states = self.context.metadata_vec(source.len())?;
        for ((_, policy), cache) in policies.iter().zip(source) {
            let LayerCachePolicy::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
                ..
            } = policy
            else {
                return Err(self.mismatch());
            };
            states.push(cache.inner().project_workspace_state(
                self.batch,
                *latent_dim,
                *rotary_dim,
                &mut projection,
            )?);
        }
        let storage = projection.try_into_storage()?;
        Ok(ProjectedPredictionLane {
            state: ProjectedPredictionState::Sequential(states),
            storage,
        })
    }

    fn pooling(
        &mut self,
        source: &[OwnedPredictionCache<MlxPoolingAttentionCache>],
    ) -> Result<ProjectedPredictionLane, Error> {
        let PredictionStateSourceLayout::Pooling(policies) = &self.layout else {
            return Err(self.mismatch());
        };
        if policies.len() != source.len() {
            return Err(self.mismatch());
        }
        let mut projection = ExistingArrayProjection::new(self.context);
        let factory = WorkspacePoolingStateFactory::new(self.batch, self.context)?;
        let mut states = self.context.metadata_vec(source.len())?;
        for ((ordinal, policy), cache) in policies.iter().zip(source) {
            states.push(cache.inner().project_workspace_layer(
                *ordinal,
                policy,
                &factory,
                self.context,
                |array| projection.project(array),
            )?);
        }
        let storage = projection.try_into_storage()?;
        Ok(ProjectedPredictionLane {
            state: ProjectedPredictionState::Pooling(states),
            storage,
        })
    }

    fn model(&mut self, source: &MlxHybridState) -> Result<ProjectedPredictionLane, Error> {
        let PredictionStateSourceLayout::Model(layout) = &self.layout else {
            return Err(self.mismatch());
        };
        if source.layout() != *layout {
            return Err(self.mismatch());
        }
        let mut projection = ExistingArrayProjection::new(self.context);
        let state = source.project_workspace_state(self.batch, &mut projection)?;
        let storage = projection.try_into_storage()?;
        Ok(ProjectedPredictionLane {
            state: ProjectedPredictionState::Model(state),
            storage,
        })
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::super::memory_tests::{pooling, pooling_policy, sequential};
    use super::*;
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
    use eredu_core::AttentionPolicy;
    use eredu_nn::{CompressedAttentionCache, PoolingAttentionCache};
    use safemlx::{Array, Device, DeviceType, Stream};
    use std::collections::BTreeMap;

    fn storage(
        arrays: impl IntoIterator<Item = impl AsRef<Array>>,
    ) -> BTreeMap<safemlx::AllocationIdentity, u64> {
        arrays
            .into_iter()
            .map(|a| {
                let a = a.as_ref();
                a.evaluated().unwrap();
                let info = a.allocation_info().unwrap().unwrap();
                (info.identity(), info.bytes() as u64)
            })
            .collect()
    }
    fn projected_storage(
        p: &ProjectedPredictionLane,
    ) -> BTreeMap<safemlx::AllocationIdentity, u64> {
        assert!(p.storage.is_complete());
        p.storage.iter().map(|(id, bytes, _)| (id, bytes)).collect()
    }
    #[test]
    fn current_prediction_profiles_preserve_frontiers_aliases_and_declared_geometry() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let batch = NonZeroU32::new(1).unwrap();
        let policy =
            LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 3, 2).unwrap();
        let policies = [(4, policy.clone()), (7, policy)];
        let first = sequential(true, &stream);
        let expected = storage(first.inner().retained_arrays());
        // Ordinary source aliases deliberately cross member boundaries. Projection
        // must retain both logical members and count each backing only once.
        let source = vec![first.clone(), first];
        let mut factory = ProjectionFactory {
            layout: PredictionStateSourceLayout::Sequential(&policies),
            batch,
            context: &context,
        };
        let projected = factory.sequential(&source).unwrap();
        assert_eq!(projected_storage(&projected), expected);
        let ProjectedPredictionState::Sequential(rows) = projected.state else {
            panic!("sequential profile");
        };
        assert_eq!(
            rows.iter().map(|row| row.offset()).collect::<Vec<_>>(),
            [3, 3]
        );
        assert_eq!(rows[0].capacity(), source[0].inner().capacity());
        assert!(factory.pooling(&[]).is_err());
        assert!(factory.sequential(&source[..1]).is_err());
        let invalid = [(
            4,
            LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 9, 2).unwrap(),
        )];
        factory.layout = PredictionStateSourceLayout::Sequential(&invalid);
        assert!(factory.sequential(&source[..1]).is_err());
        // Empty source obtains widths from the selected declaration, not arrays.
        let empty = [sequential(false, &stream)];
        factory.layout = PredictionStateSourceLayout::Sequential(&policies[..1]);
        let projected = factory.sequential(&empty).unwrap();
        assert!(projected_storage(&projected).is_empty());
        let ProjectedPredictionState::Sequential(rows) = projected.state else {
            unreachable!()
        };
        assert_eq!(rows[0].offset(), 0);

        let policies = [(0, pooling_policy()), (1, pooling_policy())];
        let first = pooling(true, &stream);
        let expected = storage(first.inner().retained_arrays());
        let source = vec![first.clone(), first];
        factory.layout = PredictionStateSourceLayout::Pooling(&policies);
        let projected = factory.pooling(&source).unwrap();
        assert_eq!(projected_storage(&projected), expected);
        let ProjectedPredictionState::Pooling(rows) = projected.state else {
            panic!("pooling profile");
        };
        assert_eq!(
            rows.iter().map(|row| row.offset()).collect::<Vec<_>>(),
            [5, 5]
        );
        assert_eq!(rows[0].pooling_ratio(0), Some(2));
        assert_eq!(rows[1].pooling_ratio(1), Some(3));
    }
}
