//! Selected conversion before the residency manager retains the resulting source.
use super::*;
use crate::backend::runtime::{
    checkpoint::bounded_quantization::{ColdConversion, ConvertedQuantization, CpuTileResources},
    execution::layerwise::plan_exact_quantization_from_destinations,
};
use eredu_checkpoint::{
    recipe::EncodedRecipeKeysPlan,
    store::{MemoryEncodedReadPlan, RetainedCheckpointSource, SafetensorsEncodedReadPlan},
    WeightQuantization,
};
use eredu_runtime::{working_memory::DependencyMemoryPolicy, ReplicatedTextMaterializationTask};
use std::collections::BTreeMap;

pub(super) fn prepare(
    source: &RetainedCheckpointSource,
    modules: &[&BTreeMap<String, eredu_nn::workspace::WorkspaceLayout>],
    tasks: &[ReplicatedTextMaterializationTask],
    pool: &MemoryLedger,
    execution_stream: &Stream,
) -> Result<Option<(RetainedCheckpointSource, Vec<ConvertedQuantization>)>, Error> {
    let groups = eredu_runtime::group_replicated_text_transform_tasks(tasks)
        .map_err(|cause| Error::Quantization(cause.to_string()))?;
    if groups.is_empty() {
        return Ok(Some((source.clone(), Vec::new())));
    }
    if safemlx::StreamCopyPlan::<()>::capture(execution_stream)
        .map_err(|cause| Error::Other(Box::new(cause)))?
        .device_type()
        != safemlx::DeviceType::Cpu
        || groups
            .iter()
            .any(|group| !matches!(group.quantization(), WeightQuantization::Affine(_)))
        || !pool.same_ledger(&crate::backend::managed_memory::ledger())
        || source.materialization_input_bytes().is_none()
    {
        return Ok(None);
    }

    let mut plans = Vec::with_capacity(groups.len());
    for group in groups {
        let selected = group
            .tasks(tasks)
            .map_err(|cause| Error::Quantization(cause.to_string()))?;
        let (plan, destinations) = plan_exact_quantization_from_destinations(
            source,
            modules,
            None,
            group.quantization(),
            &selected,
        )?;
        // Decide the complete selection before creating any native resource or
        // converting a payload. This uncached probe checks recipe geometry and
        // the closed read route; it reads no tensor bytes and grants no admission.
        for target in plan.targets() {
            let recipe = target.source();
            let Some(keys) = EncodedRecipeKeysPlan::new(recipe)? else {
                return Ok(None);
            };
            let keys = keys
                .construct(())
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            if MemoryEncodedReadPlan::from_source(source, keys.keys())
                .map_err(|cause| Error::Other(Box::new(cause)))?
                .is_none()
                && SafetensorsEncodedReadPlan::from_source(source, keys.keys())
                    .map_err(|cause| Error::Other(Box::new(cause)))?
                    .is_none()
            {
                return Ok(None);
            }
            if recipe
                .prepare_encoded_read_uncached(source.as_ref())?
                .is_none()
            {
                return Ok(None);
            }
        }
        plans.push((plan, destinations));
    }

    let resources = CpuTileResources::prepare(pool)
        .map_err(|cause| Error::Other(Box::new(cause.into_backend_failure())))?;
    let mut store = source.clone();
    let mut completed = Vec::with_capacity(plans.len());
    for (plan, destinations) in plans {
        let converted = ColdConversion {
            source: &store,
            plan: &plan,
            pool,
            resources: &resources,
            metadata_policy: DependencyMemoryPolicy::default(),
            destinations: Some(&destinations),
        }
        .prepare()
        .map_err(|cause| Error::Other(Box::new(cause)))?;
        store = converted.store().clone();
        completed.push(converted);
    }
    Ok(Some((store, completed)))
}
