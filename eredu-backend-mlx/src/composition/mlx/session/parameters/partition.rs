//! Public loaded parameter access over actual prepared partition owners.
use super::*;
use eredu_runtime::parameter_operations::*;
use sha2::{Digest, Sha256};
#[path = "partition/overlay.rs"]
mod overlay;
#[path = "partition/read.rs"]
mod read;

pub(super) struct GlobalParameterCatalog {
    pub(super) discovery: ParameterDiscovery,
    pub(super) layouts: BTreeMap<String, EffectiveLayout>,
    global: PartitionParameterCatalog,
}
struct LocalCatalog {
    artifact: String,
    layouts: BTreeMap<String, EffectiveLayout>,
    facts: Vec<LocalParameterFacts>,
    geometry: [u8; 32],
    max_bytes: usize,
}

fn parameter_outputs<'a>(
    tasks: impl Iterator<Item = &'a eredu_runtime::ReplicatedTextMaterializationTask>,
) -> impl Iterator<Item = (&'a str, &'a [usize], eredu_checkpoint::LinearFormat)> {
    tasks.flat_map(|task| {
        std::iter::once(task.name())
            .chain(task.aliases().iter().map(String::as_str))
            .map(move |name| (name, task.logical_shape(), task.executable()))
            .chain(task.output_companions().iter().map(|companion| {
                (
                    companion.name(),
                    companion.logical_shape(),
                    eredu_checkpoint::LinearFormat::Dense,
                )
            }))
    })
}
const UNLIMITED: CaptureUsage = CaptureUsage {
    captures: u64::MAX,
    retained_bytes: u64::MAX,
    host_bytes: u64::MAX,
    encoded_bytes: u64::MAX,
};
const DISCOVERY_LIMIT: CaptureUsage = CaptureUsage {
    captures: 1,
    retained_bytes: 256 << 20,
    host_bytes: 1 << 30,
    encoded_bytes: 128 << 20,
};

impl MlxModelSession {
    pub(super) fn prepared_parameter_transport<'a>(
        &self,
        transport: &'a crate::backend::distributed::MlxDistributedSession,
    ) -> Result<crate::backend::distributed::PreparedParameterTransport<'a>, ParameterError> {
        let ledger = &self.payload.memory_ledger;
        crate::backend::distributed::PreparedParameterTransport::prepare(
            transport,
            ledger,
            self.payload.model.erased().inference_execution_identity(),
            ledger.configured_limits().clone(),
        )
        .map_err(failure)
    }

    pub(super) fn partition_parameter_catalog(
        &mut self,
        limits: Option<CaptureUsage>,
    ) -> Result<GlobalParameterCatalog, ParameterError> {
        let transport = self.payload.distributed.clone().ok_or_else(|| {
            ParameterError::Unsupported("partition has no retained communication owner".into())
        })?;
        let owner = transport.parameter_operations()?;
        let prepared_transport = self.prepared_parameter_transport(&transport)?;
        let binding = self
            .payload
            .parameter_state
            .model_identity
            .as_ref()
            .ok_or_else(|| {
                ParameterError::Unsupported("partition has no loaded parameter identity".into())
            })?
            .catalog_binding(self.payload.parameter_state.epoch);
        let mut budget = NativeParameterBudget {
            total: Rc::clone(&self.payload.parameter_state.usage),
            limit: limits.unwrap_or(UNLIMITED),
            operation: limits
                .is_none()
                .then(|| (Rc::new(Cell::new(CaptureUsage::default())), DISCOVERY_LIMIT)),
        };
        let prepared = self.prepare_local_parameter_catalog(
            transport.parameter_rank(),
            transport.parameter_setup().participant_count(),
            &mut budget,
        );
        let (local, admission) = match prepared {
            Ok(local) => {
                let admission = ParameterReadPreparation::new(
                    (),
                    local.geometry,
                    local.max_bytes.div_ceil(4) + 2,
                );
                (Some(local), Ok(admission))
            }
            Err(error) => (None, Err(error)),
        };
        let mut encoding_budget = budget.clone();
        let mut assembly_budget = budget.clone();
        let result = owner.read(
            &prepared_transport,
            binding,
            &[0x63; 32],
            ParameterOperationKind::Catalog,
            admission,
            &mut budget,
            |_| {
                let local = local.as_ref().expect("admitted local catalogue");
                encode_parameter_catalog(&local.facts, local.max_bytes, &mut encoding_budget)
            },
            |rows| {
                let local = local.as_ref().expect("admitted local catalogue");
                assembly_budget.reserve_quota(CaptureUsage {
                    host_bytes: mul(rows.len() as u64, 64)?,
                    ..Default::default()
                })?;
                let facts = rows
                    .iter()
                    .map(|row| decode_parameter_catalog(row, local.max_bytes, &mut assembly_budget))
                    .collect::<Result<Vec<_>, _>>()?;
                let prepared = self
                    .capture_discovery
                    .as_ref()
                    .expect("admitted prepared discovery");
                PartitionParameterCatalog::new(
                    &facts,
                    |rank, parameter, budget| {
                        // Sharing is supplied by actual prepared slot metadata.
                        // Distinct tasks can intentionally share a module parameter;
                        // runtime validates the canonical class and matching shape.
                        let layout = prepared
                            .parameter_partition_layout_for_rank(&parameter.id, rank, budget)?
                            .ok_or_else(|| {
                                ParameterError::Unsupported(
                                    "retained parameter placement is unavailable".into(),
                                )
                            })?;
                        if layout.global_shape() != parameter.shape {
                            return Err(ParameterError::Invalid(
                                "loaded global parameter shape differs from retained task".into(),
                            ));
                        }
                        Ok(layout.into_coordinates())
                    },
                    &mut assembly_budget,
                )
            },
        );
        let global = self.finish_parameter_control(&transport, result)?;
        let local = local.expect("completed catalogue preparation");
        // The initial metadata allowance includes this public descriptor copy
        // as well as the local maps/facts; no parameter tensor is retained.
        let discovery = ParameterDiscovery {
            identity: format!(
                "{}:parameters:{}",
                self.intervention_session_identity, self.payload.parameter_state.epoch
            ),
            artifact_identity: local.artifact,
            overlay_identity: self.payload.parameter_state.active.clone(),
            parameters: global.parameters().to_vec(),
            usage: self.payload.parameter_state.usage.get(),
            coordination_usage: owner.usage()?,
        };
        Ok(GlobalParameterCatalog {
            discovery,
            layouts: local.layouts,
            global,
        })
    }

    fn prepare_local_parameter_catalog(
        &self,
        rank: usize,
        world: usize,
        budget: &mut NativeParameterBudget,
    ) -> Result<LocalCatalog, ParameterError> {
        self.ensure_no_submission_in_flight().map_err(failure)?;
        let prepared = self.capture_discovery.as_ref().ok_or_else(|| {
            ParameterError::Unsupported("no retained prepared parameter identity".into())
        })?;
        let tasks = prepared.partition_parameter_tasks().ok_or_else(|| {
            ParameterError::Unsupported("no global prepared parameter declaration".into())
        })?;
        let shared_name_bytes = parameter_outputs(tasks.clone())
            .map(|(name, _, _)| name.len())
            .max()
            .unwrap_or(0) as u64;
        let mut max_bytes = 1024u64;
        for (id, shape, _) in parameter_outputs(tasks.clone()) {
            max_bytes = add(
                max_bytes,
                add(
                    1024,
                    add(
                        mul(add(id.len() as u64, shared_name_bytes)?, 6)?,
                        mul(shape.len() as u64, 64)?,
                    )?,
                )?,
            )?;
        }
        // The native metadata protocol has finite per-rank and padded-world bounds.
        if max_bytes > 16 << 20 || mul(max_bytes, world as u64)? > 128 << 20 {
            return Err(CaptureError::Limit {
                budget: CaptureBudget::Host,
                cumulative: false,
            }
            .into());
        }
        budget.reserve_quota(CaptureUsage {
            captures: 1,
            // Include the temporary index of borrowed task names and formats.
            host_bytes: mul(max_bytes, 9)?,
            encoded_bytes: max_bytes,
            ..Default::default()
        })?;
        let artifact = prepared.capture()?.artifact_identity;
        let mut digest = Sha256::new();
        digest.update(b"eredu-loaded-parameter-catalog-v1\0");
        for value in [&artifact, prepared.execution_identity()] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        let slots = self.payload.model.erased().prepared_parameter_slots();
        let mut output_formats = BTreeMap::new();
        for (name, _, format) in parameter_outputs(tasks.clone()) {
            // Match the previous first-output lookup. Retained placement still
            // rejects any ambiguous selected identity before publishing facts.
            output_formats.entry(name).or_insert(format);
        }
        let mut catalog = Catalog {
            layouts: BTreeMap::new(),
            companions: BTreeSet::new(),
            physical: BTreeMap::new(),
            slots: Vec::new(),
        };
        let mut global_shapes = BTreeMap::new();
        for slot in slots {
            let id = slot.parameter.id.as_str();
            let format = output_formats
                .get(id)
                .copied()
                .ok_or_else(|| ParameterError::Missing(id.into()))?;
            let placement = prepared
                .parameter_partition_layout_for_rank(id, rank, budget)?
                .ok_or_else(|| {
                    ParameterError::Unsupported("no effective local placement".into())
                })?;
            let coordinates = placement.coordinates().ok_or_else(|| {
                ParameterError::Invalid("prepared slot has no selected rank ownership".into())
            })?;
            catalog.layouts.insert(
                id.into(),
                EffectiveLayout::new(format, coordinates.local_shape().to_vec()),
            );
            global_shapes.insert(id.to_owned(), placement.global_shape().to_vec());
        }
        for slot in slots {
            let id = slot.parameter.id.as_str();
            let (physical_dtype, shape) =
                if let Some(original) = self.payload.parameter_state.originals.get(id) {
                    let dtype = if dtype(original.as_array().dtype()).is_some() {
                        original.as_array().dtype()
                    } else {
                        Dtype::Float32
                    };
                    (dtype, catalog.layouts[id].shape.clone())
                } else {
                    (
                        crate::backend::runtime::checkpoint::recipe::mlx_dtype(
                            &slot.materialized.dtype,
                        )
                        .map_err(|error| ParameterError::Unsupported(error.to_string()))?,
                        slot.materialized.shape.iter().map(|n| *n as u64).collect(),
                    )
                };
            catalog.record(slot.parameter.clone(), physical_dtype, shape);
        }
        catalog.bind_bank_loan_costs(slots)?;
        catalog.finish(&self.payload.parameter_state.originals)?;
        let state_reset = self
            .payload
            .model
            .erased()
            .estimate_parameter_reset_state()
            .is_some();
        let facts = catalog
            .slots
            .into_iter()
            .map(|mut parameter| {
                let local_shape = parameter.shape.clone();
                parameter.shape = global_shapes
                    .remove(&parameter.id)
                    .expect("prepared global shape");
                parameter.access = Some(ParameterAccess {
                    query: parameter.supported,
                    projection: parameter.supported,
                    replacement: parameter.supported && state_reset,
                });
                if parameter.supported && !state_reset {
                    parameter.condition.push_str("; distributed replacement requires a bounded fresh-state exchange mechanism");
                }
                parameter.supported &= state_reset;
                LocalParameterFacts {
                    parameter,
                    local_shape,
                }
            })
            .collect();
        Ok(LocalCatalog {
            artifact,
            layouts: catalog.layouts,
            facts,
            geometry: digest.finalize().into(),
            max_bytes: usize::try_from(max_bytes).map_err(|_| ParameterError::Overflow)?,
        })
    }

    fn finish_parameter_control<T>(
        &mut self,
        transport: &MlxDistributedSession,
        result: Result<T, ParameterError>,
    ) -> Result<T, ParameterError> {
        if result.is_err() && transport.ensure_parameter_active().is_err() {
            self.poison.set(true);
        }
        result
    }
}
