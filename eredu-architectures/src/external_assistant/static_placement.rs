//! Source-free static target placement from retained effective parameter coordinates.
use crate::{
    composite_execution::ExternalPredictionTargetOperationKind as Kind,
    prepared_sources::PreparedModelDiscovery,
};
use eredu_core::{
    capture::{CaptureBudget, CaptureError, CaptureReservation, CaptureSkipReason, CaptureUsage},
    ParallelRankTopology,
};
use eredu_nn::Error;
use eredu_runtime::{
    CommunicationGroupRequirements, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationTensorLimits, PredictionTensorAssembly, PredictionTensorAssemblyPlan,
    PredictionTensorSource,
};
use std::{collections::BTreeMap, ops::Range};

#[derive(Debug, Clone)]
pub(crate) struct TargetStaticDescription {
    embedding: String,
    projection: String,
    norm: String,
    hidden: i32,
    vocabulary: i32,
}
impl TargetStaticDescription {
    pub(crate) fn from_plan(
        plan: &crate::processor_plan::ArtifactArchitecturePlan,
    ) -> Result<Option<Self>, String> {
        match crate::replicated_text::composite_config(plan).map_err(|e| e.to_string())? {
            Some(crate::replicated_text::CompositeConfig::Muse(args)) => {
                Ok(Some(Self::muse(&args)))
            }
            _ => Ok(None),
        }
    }
    pub(super) fn muse(args: &crate::muse::DecoderConfig) -> Self {
        Self {
            embedding: "model.embed_tokens.weight".into(),
            projection: if args.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
            .into(),
            norm: "model.norm.weight".into(),
            hidden: args.hidden_size,
            vocabulary: args.vocab_size,
        }
    }

    pub(super) fn compute_precisions(
        &self,
        kind: Kind,
        tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
    ) -> Vec<eredu_core::checkpoint::TensorDtype> {
        let selected = tasks
            .iter()
            .filter(|task| {
                let matches = |name: &str| {
                    task.name() == name || task.aliases().iter().any(|alias| alias == name)
                };
                if kind.is_lookup() {
                    matches(&self.embedding)
                } else {
                    matches(&self.projection)
                        || (kind == Kind::ProjectLogits && matches(&self.norm))
                }
            })
            .cloned()
            .collect::<Vec<_>>();
        crate::linear_format::floating_compute_precisions(&selected)
    }
}

/// Immutable authoritative vocabulary ranges for one target-owned static equation.
/// Replicated tables are storage, rather than additional additive contributions.
#[derive(Debug, Clone)]
pub struct PreparedTargetStaticPlacement {
    topology: ParallelRankTopology,
    identity: String,
    kind: Kind,
    hidden: i32,
    vocabulary: i32,
    sources: Vec<(usize, Range<usize>)>,
    publication: CommunicationGroupRequirements,
}
impl PreparedTargetStaticPlacement {
    /// Rank authority retained from the selected execution.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.topology
    }
    /// Resolves exact input geometry without reading tensors or artifact payloads.
    pub fn resolve(
        &self,
        kind: Kind,
        input_shape: &[i32],
    ) -> Result<PredictionTensorAssemblyPlan, Error> {
        let limits = self.publication.operations()[0]
            .limits()
            .expect("prepared static publication has tensor limits");
        if kind != self.kind {
            return Err(Error::backend(
                "static placement equation differs from requested operation",
            ));
        }
        let (batch, sequence) = match (kind.is_lookup(), input_shape) {
            (true, [batch, sequence]) => (*batch, *sequence),
            (false, [batch, sequence, hidden]) if *hidden == self.hidden => (*batch, *sequence),
            _ => {
                return Err(Error::backend(
                    "target static input rank or hidden width differs",
                ))
            }
        };
        if batch <= 0 || sequence <= 0 {
            return Err(Error::backend(
                "target static input has a nonpositive extent",
            ));
        }
        let width = if kind.is_lookup() {
            self.hidden
        } else {
            self.vocabulary
        };
        let complete = checked_elements(&[batch, sequence, width])?;
        if limits.max_tensor_rank() < 3 || complete > limits.max_output_tensor_elements() {
            return Err(Error::backend(
                "target static invocation exceeds retained publication geometry",
            ));
        }
        let sources = self
            .sources
            .iter()
            .map(|(rank, rows)| {
                let width = if kind.is_lookup() {
                    self.hidden
                } else {
                    i32::try_from(rows.len()).map_err(Error::backend_source)?
                };
                let shape = vec![batch, sequence, width];
                if checked_elements(&shape)? > limits.max_tensor_elements() {
                    return Err(Error::backend(
                        "target static contribution exceeds retained publication geometry",
                    ));
                }
                PredictionTensorSource::new(*rank, shape)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        PredictionTensorAssemblyPlan::new(
            &self.identity,
            sources,
            if kind.is_lookup() {
                PredictionTensorAssembly::Sum
            } else {
                PredictionTensorAssembly::Concatenate { axis: 2 }
            },
        )
    }

    pub(super) fn resolve_admitted(
        &self,
        kind: Kind,
        input_shape: &[i32],
        precisions: Vec<eredu_core::checkpoint::TensorDtype>,
    ) -> Result<PredictionTensorAssemblyPlan, Error> {
        let plan = self.resolve(kind, input_shape)?;
        let input = plan
            .sources()
            .iter()
            .map(|source| checked_elements(source.shape()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| Error::backend("static invocation has no publication sources"))?;
        let output = checked_elements(plan.shape())?;
        let limits = CommunicationTensorLimits::new(1, 3, input, None)
            .and_then(|limits| limits.with_output_tensor_elements(output))
            .map_err(Error::backend_source)?;
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            precisions,
            limits,
            true,
        )
        .map_err(Error::backend_source)?;
        self.publication
            .validate_requirement(&requirement)
            .map_err(Error::backend_source)?;
        Ok(plan)
    }
    pub(crate) fn prepare(
        discovery: &PreparedModelDiscovery,
        description: &TargetStaticDescription,
        kind: Kind,
    ) -> Result<Option<Self>, Error> {
        let Some(selected) = discovery.static_partition_selection() else {
            return Ok(None);
        };
        Self::prepare_using(
            selected,
            description,
            kind,
            discovery.execution_identity(),
            |parameter, rank, reservation| {
                discovery
                    .parameter_partition_layout_for_rank(parameter, rank, reservation)
                    .map_err(Error::backend_source)
            },
        )
    }

    /// Uses the same effective-coordinate calculation before native construction.
    /// Its identity is descriptive and grants no session or transport authority.
    pub(super) fn prepare_cold(
        selected: &crate::SelectedExecution,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        index: &crate::parameter_partition::ParameterMemberIndex,
        description: &TargetStaticDescription,
        kind: Kind,
    ) -> Result<Option<Self>, Error> {
        Self::prepare_using(
            selected,
            description,
            kind,
            "cold-target-static-admission",
            |parameter, rank, reservation| {
                selected
                    .parameter_partition_layout_with_index(
                        parameters,
                        parameter,
                        rank,
                        Some(index),
                        reservation,
                    )
                    .map_err(Error::backend_source)
            },
        )
    }

    fn prepare_using(
        selected: &crate::SelectedExecution,
        description: &TargetStaticDescription,
        kind: Kind,
        execution_identity: &str,
        mut parameter_layout: impl FnMut(
            &str,
            usize,
            &mut MetadataReservation,
        ) -> Result<
            Option<crate::parameter_partition::ParameterPartitionLayout>,
            Error,
        >,
    ) -> Result<Option<Self>, Error> {
        let topology = selected
            .parallel_topology()
            .ok_or_else(|| Error::backend("static selection has no rank authority"))?;
        let group = selected
            .partitioned_session_group()
            .ok_or_else(|| Error::backend("static selection has no session publication group"))?;
        let publication = selected
            .communication_manifest()
            .and_then(|manifest| manifest.groups().iter().find(|g| g.id() == group))
            .and_then(|g| {
                g.requirements()
                    .operations()
                    .iter()
                    .find(|op| op.operation() == CommunicationOperation::Broadcast)
            })
            .ok_or_else(|| {
                Error::backend("static selection has no retained broadcast requirement")
            })?;
        if !publication
            .dtypes()
            .contains(&eredu_core::checkpoint::TensorDtype::F32)
        {
            return Err(Error::backend(
                "static publication does not admit the precision header",
            ));
        }
        publication
            .limits()
            .ok_or_else(|| Error::backend("static publication has no tensor bounds"))?;
        let publication = CommunicationGroupRequirements::new([publication.clone()])
            .map_err(Error::backend_source)?;
        let budget = (topology.world_size() as u64)
            .checked_mul(1 << 20)
            .ok_or_else(|| Error::backend("static placement metadata bound overflows"))?;
        let mut reservation = MetadataReservation { remaining: budget };
        reserve(
            &mut reservation,
            1024 + execution_identity.len() as u64 + topology.world_size() as u64 * 128,
        )?;
        let target = if kind.is_lookup() {
            &description.embedding
        } else {
            &description.projection
        };
        let mut ranges = BTreeMap::new();
        for rank in 0..topology.world_size() {
            let layout = parameter_layout(target, rank, &mut reservation)?
                .ok_or_else(|| Error::backend("static parameter placement is absent"))?;
            if layout.global_shape() != [description.vocabulary as u64, description.hidden as u64] {
                return Err(Error::backend(
                    "static table effective geometry differs from normalized architecture",
                ));
            }
            let Some(coordinates) = layout.coordinates() else {
                continue;
            };
            let axes = coordinates.axes();
            let rows = axes[0]
                .contiguous_range()
                .ok_or_else(|| Error::backend("static vocabulary rows are not contiguous"))?;
            if axes[1].contiguous_range() != Some(0..description.hidden as usize) || rows.is_empty()
            {
                return Err(Error::backend(
                    "static table does not own a complete hidden axis",
                ));
            }
            if kind == Kind::ProjectLogits {
                let norm = parameter_layout(&description.norm, rank, &mut reservation)?
                    .ok_or_else(|| Error::backend("static final norm placement is absent"))?;
                if norm.global_shape() != [description.hidden as u64] {
                    return Err(Error::backend("static final norm geometry differs"));
                }
                let Some(norm) = norm.coordinates() else {
                    continue;
                };
                if norm.axes()[0].contiguous_range() != Some(0..description.hidden as usize) {
                    return Err(Error::backend(
                        "static final norm is not complete on the projection source",
                    ));
                }
            }
            // Iteration is in world-rank order; identical physical replicas do
            // not change the authoritative source or the addition coefficient.
            ranges.entry((rows.start, rows.end)).or_insert(rank);
        }
        let mut cursor = 0;
        let mut sources = Vec::with_capacity(ranges.len());
        for ((start, end), rank) in ranges {
            if start != cursor || end <= start {
                return Err(Error::backend(
                    "static vocabulary sources overlap or leave a gap",
                ));
            }
            cursor = end;
            sources.push((rank, start..end));
        }
        if cursor != description.vocabulary as usize {
            return Err(Error::backend(
                "static vocabulary sources do not cover the complete table",
            ));
        }
        let mut identity = format!("{execution_identity}:target-static:{kind:?}:sources=");
        for (rank, rows) in &sources {
            use std::fmt::Write as _;
            write!(&mut identity, "{rank}:{}:{};", rows.start, rows.end)
                .expect("writing to a String is infallible");
        }
        Ok(Some(Self {
            topology,
            identity,
            kind,
            hidden: description.hidden,
            vocabulary: description.vocabulary,
            sources,
            publication,
        }))
    }
}
fn checked_elements(shape: &[i32]) -> Result<usize, Error> {
    shape
        .iter()
        .try_fold(1usize, |n, d| {
            usize::try_from(*d).ok().and_then(|d| n.checked_mul(d))
        })
        .ok_or_else(|| Error::backend("static invocation element count overflows"))
}
struct MetadataReservation {
    remaining: u64,
}
impl CaptureReservation for MetadataReservation {
    fn reserve(&mut self, usage: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        if usage.captures != 0 || usage.retained_bytes != 0 || usage.encoded_bytes != 0 {
            return Err(CaptureError::Invalid(
                "static placement may reserve host metadata only".into(),
            ));
        }
        self.remaining =
            self.remaining
                .checked_sub(usage.host_bytes)
                .ok_or(CaptureError::Limit {
                    budget: CaptureBudget::Host,
                    cumulative: false,
                })?;
        Ok(None)
    }
}
fn reserve(budget: &mut MetadataReservation, bytes: u64) -> Result<(), Error> {
    budget
        .reserve(CaptureUsage {
            host_bytes: bytes,
            ..Default::default()
        })
        .map_err(Error::backend_source)?;
    Ok(())
}
