//! Rank-local target geometry and exact physical parameter selections.
use super::*;
use eredu_checkpoint::store::TensorSelection;
use eredu_nn::{GatedProductGroupLayout, GroupedProjectionSpec, LogicalParameterDeclaration};
use eredu_runtime::{MemberSharding, ParameterRole};
use std::ops::Range;

#[path = "parallel_layout.rs"]
pub(crate) mod parallel_layout;

/// One tensor rank's local mixer/FFN geometry. Vocabulary, residual mixers,
/// lexical injection, routers and QSA indexers remain explicitly replicated.
#[derive(Debug, Clone)]
pub struct TargetTensorPartition {
    rank: usize,
    ranks: usize,
    source_identity: String,
    local: TargetSpec,
    selections: BTreeMap<String, Vec<TensorSelection>>,
}
impl TargetTensorPartition {
    /// Tensor rank in the declared group.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Number of tensor ranks.
    pub const fn ranks(&self) -> usize {
        self.ranks
    }
    /// Global request policy with localized execution units and state geometry.
    pub fn local_spec(&self) -> &TargetSpec {
        &self.local
    }
    /// Ordered physical ranges; absent entries retain their complete parameter.
    /// Multiple ranges preserve component-major fused projection ordering.
    pub fn parameter_selections(&self, parameter: &str) -> Option<&[TensorSelection]> {
        self.selections.get(parameter).map(Vec::as_slice)
    }
}
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::backend(format!("qwen4_exp tensor partition: {detail}"))
}
fn divide(width: i32, ranks: usize) -> Result<i32, Error> {
    if width <= 0 || !(width as usize).is_multiple_of(ranks) {
        return Err(invalid(format!(
            "width {width} cannot divide into {ranks} ranks"
        )));
    }
    Ok(width / ranks as i32)
}
pub(crate) struct Selections(pub(crate) BTreeMap<String, Vec<TensorSelection>>);
impl Selections {
    fn raw(
        &mut self,
        name: &str,
        axis: usize,
        segments: &[Range<usize>],
        rank: usize,
        ranks: usize,
    ) -> Result<(), Error> {
        let values = segments
            .iter()
            .map(|segment| {
                if segment.is_empty() || !segment.len().is_multiple_of(ranks) {
                    return Err(invalid(format!(
                        "parameter {name} segment {segment:?} cannot divide into {ranks} ranks"
                    )));
                }
                let width = segment.len() / ranks;
                Ok(TensorSelection::Range {
                    axis,
                    start: segment.start + rank * width,
                    end: segment.start + (rank + 1) * width,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if self.0.insert(name.to_owned(), values).is_some() {
            return Err(invalid(format!("duplicate parameter {name}")));
        }
        Ok(())
    }
    fn projection(
        &mut self,
        declaration: &LogicalParameterDeclaration<'_>,
        bias: Option<&ParameterSpec>,
        axis: usize,
        segments: Vec<Range<usize>>,
        rank: usize,
        ranks: usize,
    ) -> Result<(), Error> {
        let weight = declaration.parameter;
        let shape = &declaration.shape;
        let format = declaration
            .linear_format
            .ok_or_else(|| invalid(format!("projection {} has no declared encoding", weight.id)))?;
        let row_axis = shape.len() - 2;
        if let eredu_checkpoint::LinearFormat::GgufIQuant { ggml_type, .. } = format.encoding() {
            if axis == shape.len() - 1 {
                let (values, _) = ggml_type.block_and_bytes().map_err(invalid)?;
                let values = values as usize;
                if segments.iter().any(|segment| {
                    !segment.len().is_multiple_of(ranks)
                        || !(segment.len() / ranks).is_multiple_of(values)
                }) {
                    return Err(invalid(format!(
                        "parameter {} tensor shard splits a GGUF encoding block",
                        weight.id
                    )));
                }
            }
        }
        // Retained declarations supply geometry, sharing and exact encoding companions.
        // Family semantics still decide the independently split source segments.
        let group = eredu_runtime::logical_parameter_group(
            weight.id.as_str(),
            ParameterRole::Segmented,
            None,
            &vec![declaration.clone()],
            |_| MemberSharding::Segmented {
                axis,
                segments: segments.clone(),
            },
        )
        .map_err(invalid)?;
        for member in group.members() {
            let MemberSharding::Segmented { axis, segments } = member.sharding() else {
                return Err(invalid("physical projection lost segmented placement"));
            };
            self.raw(member.target(), *axis, segments, rank, ranks)?;
        }
        if axis == row_axis {
            if let Some(bias) = bias {
                self.raw(bias.id.as_str(), row_axis, &segments, rank, ranks)?;
            }
        }
        Ok(())
    }
    fn linear(
        &mut self,
        spec: &LinearSpec,
        axis: usize,
        segments: Vec<Range<usize>>,
        rank: usize,
        ranks: usize,
    ) -> Result<(), Error> {
        let parameters = spec.logical_parameters()?;
        let declaration = parameters
            .iter()
            .find(|parameter| parameter.parameter.id == spec.weight.id)
            .ok_or_else(|| invalid(format!("undeclared projection {}", spec.weight.id)))?;
        self.projection(declaration, spec.bias.as_ref(), axis, segments, rank, ranks)
    }
    fn grouped(
        &mut self,
        spec: &GroupedProjectionSpec,
        parameters: &BTreeMap<&str, &LogicalParameterDeclaration<'_>>,
        axis: usize,
        segments: Vec<Range<usize>>,
        rank: usize,
        ranks: usize,
    ) -> Result<(), Error> {
        let declaration = parameters
            .get(spec.weight().id.as_str())
            .copied()
            .ok_or_else(|| invalid(format!("undeclared expert projection {}", spec.weight().id)))?;
        self.projection(declaration, spec.bias(), axis, segments, rank, ranks)
    }
}

impl Selections {
    #[allow(clippy::single_range_in_vec_init)]
    pub(crate) fn partition_sublayers(
        &mut self,
        mixer: &mut MixerSpec,
        feed_forward: &mut FeedForwardSublayerSpec,
        rank: usize,
        ranks: usize,
    ) -> Result<(), Error> {
        if ranks == 0 || ranks > i32::MAX as usize || rank >= ranks {
            return Err(invalid("invalid rank or group size"));
        }
        if ranks == 1 {
            return Ok(());
        }
        match mixer {
            MixerSpec::Recurrent(s) => {
                let r = &mut s.mixer;
                let segments = r.geometry.qkv_ranges()?.to_vec();
                self.linear(&r.input_qkv, 0, segments.clone(), rank, ranks)?;
                self.raw(r.convolution.weight.id.as_str(), 0, &segments, rank, ranks)?;
                if let Some(bias) = &r.convolution.bias {
                    self.raw(bias.id.as_str(), 0, &segments, rank, ranks)?;
                }
                for projection in [&r.input_gate, &r.input_beta, &r.input_decay] {
                    self.linear(
                        projection,
                        0,
                        vec![0..projection.output as usize],
                        rank,
                        ranks,
                    )?;
                }
                self.linear(&r.output, 1, vec![0..r.output.input as usize], rank, ranks)?;
                for parameter in [&r.decay_bias, &r.transition_log] {
                    self.raw(
                        parameter.id.as_str(),
                        0,
                        &[0..r.geometry.value_heads() as usize],
                        rank,
                        ranks,
                    )?;
                }
                r.geometry = eredu_nn::GatedDeltaHeadGeometry::new(
                    divide(r.geometry.key_heads(), ranks)?,
                    divide(r.geometry.value_heads(), ranks)?,
                    r.geometry.key_dimensions(),
                    r.geometry.value_dimensions(),
                )?;
                r.input_qkv.output = divide(r.input_qkv.output, ranks)?;
                r.input_gate.output = divide(r.input_gate.output, ranks)?;
                r.input_beta.output = divide(r.input_beta.output, ranks)?;
                r.input_decay.output = divide(r.input_decay.output, ranks)?;
                r.output.input = divide(r.output.input, ranks)?;
                r.convolution.channels = divide(r.convolution.channels, ranks)?;
                s.validate()?;
            }
            MixerSpec::Indexed(s) => {
                let kv = s.parameters.projection.geometry.key_value_heads() as usize;
                let (kv_rank, kv_ranks) = if kv >= ranks {
                    if !kv.is_multiple_of(ranks) {
                        return Err(invalid("K/V heads do not divide tensor ranks"));
                    }
                    (rank, ranks)
                } else {
                    if !ranks.is_multiple_of(kv) {
                        return Err(invalid("tensor ranks do not replicate K/V heads evenly"));
                    }
                    (rank / (ranks / kv), kv)
                };
                self.linear(
                    &s.parameters.projection.query,
                    0,
                    vec![0..s.parameters.projection.query.output as usize],
                    rank,
                    ranks,
                )?;
                for p in [&s.parameters.projection.key, &s.parameters.projection.value] {
                    self.linear(p, 0, vec![0..p.output as usize], kv_rank, kv_ranks)?;
                }
                self.linear(
                    &s.parameters.projection.output,
                    1,
                    vec![0..s.parameters.projection.output.input as usize],
                    rank,
                    ranks,
                )?;
                let geometry = s.parameters.projection.geometry;
                s.parameters.projection.geometry = eredu_nn::AttentionHeadGeometry::new(
                    divide(geometry.query_heads(), ranks)?,
                    divide(geometry.key_value_heads(), kv_ranks)?,
                    geometry.key_dimensions(),
                    geometry.value_dimensions(),
                )?;
                s.parameters.projection.query.output =
                    divide(s.parameters.projection.query.output, ranks)?;
                s.parameters.projection.key.output =
                    divide(s.parameters.projection.key.output, kv_ranks)?;
                s.parameters.projection.value.output =
                    divide(s.parameters.projection.value.output, kv_ranks)?;
                s.parameters.projection.output.input =
                    divide(s.parameters.projection.output.input, ranks)?;
                s.validate().map_err(Error::backend_source)?;
            }
        }
        let ff = &mut feed_forward.feed_forward;
        let expert_parameters = ff.parameters.routed.experts.logical_parameters()?;
        let expert_declarations = expert_parameters
            .iter()
            .map(|parameter| (parameter.parameter.id.as_str(), parameter))
            .collect::<BTreeMap<_, _>>();
        let intermediate = ff.parameters.routed.experts.intermediate_dimensions() as usize;
        match ff.parameters.routed.experts.layout() {
            GatedProductGroupLayout::Packed { gate_up, down } => {
                self.grouped(
                    gate_up,
                    &expert_declarations,
                    1,
                    vec![0..intermediate, intermediate..2 * intermediate],
                    rank,
                    ranks,
                )?;
                self.grouped(
                    down,
                    &expert_declarations,
                    2,
                    vec![0..intermediate],
                    rank,
                    ranks,
                )?;
            }
            GatedProductGroupLayout::Independent(experts) => {
                for expert in experts {
                    for projection in [expert.gate(), expert.up()] {
                        self.grouped(
                            projection,
                            &expert_declarations,
                            0,
                            vec![0..intermediate],
                            rank,
                            ranks,
                        )?;
                    }
                    self.grouped(
                        expert.down(),
                        &expert_declarations,
                        1,
                        vec![0..intermediate],
                        rank,
                        ranks,
                    )?;
                }
            }
            _ => return Err(invalid("unsupported expert layout")),
        }
        ff.parameters.routed.experts = ff.parameters.routed.experts.clone().with_group_geometry(
            ff.parameters.routed.experts.group_count(),
            divide(
                ff.parameters.routed.experts.intermediate_dimensions(),
                ranks,
            )?,
        )?;
        for p in [&ff.parameters.shared.gate, &ff.parameters.shared.up] {
            self.linear(p, 0, vec![0..p.output as usize], rank, ranks)?;
        }
        self.linear(
            &ff.parameters.shared.down,
            1,
            vec![0..ff.parameters.shared.down.input as usize],
            rank,
            ranks,
        )?;
        ff.parameters.shared.gate.output = divide(ff.parameters.shared.gate.output, ranks)?;
        ff.parameters.shared.up.output = divide(ff.parameters.shared.up.output, ranks)?;
        ff.parameters.shared.down.input = divide(ff.parameters.shared.down.input, ranks)?;
        feed_forward.validate()?;
        Ok(())
    }
}

impl TargetSpec {
    /// Derives local equations and physical source transforms without source reads.
    /// Both grouped-query K/V replication and fused component boundaries are explicit.
    pub fn tensor_partition(
        &self,
        rank: usize,
        ranks: usize,
    ) -> Result<TargetTensorPartition, Error> {
        if ranks == 0 || ranks > i32::MAX as usize || rank >= ranks {
            return Err(invalid("invalid rank or group size"));
        }
        self.state_layout()?;
        if ranks == 1 {
            return Ok(TargetTensorPartition {
                rank,
                ranks,
                source_identity: self.geometry_fingerprint(),
                local: self.clone(),
                selections: BTreeMap::new(),
            });
        }
        let mut local = self.clone();
        let mut selections = Selections(BTreeMap::new());
        for unit in &mut local.units {
            let UnitSpec::Decoder {
                mixer,
                feed_forward,
                ..
            } = unit
            else {
                continue;
            };
            selections.partition_sublayers(mixer, feed_forward, rank, ranks)?;
        }
        local.state_layout()?;
        Ok(TargetTensorPartition {
            rank,
            ranks,
            source_identity: self.geometry_fingerprint(),
            local,
            selections: selections.0,
        })
    }
}
impl BoundTargetSpec {
    /// Preserves exact hash constants while deriving a rank-specific state identity.
    pub fn tensor_partition(&self, partition: &TargetTensorPartition) -> Result<Self, Error> {
        if self.spec.geometry_fingerprint() != partition.source_identity {
            return Err(invalid("partition belongs to another target geometry"));
        }
        let mut local = Self::new(partition.local.clone(), self.hashes.clone())?;
        local.state_fingerprint = eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "qwen4_exp.target.tensor_partition.v1",
            [
                ("source", self.state_fingerprint.clone()),
                ("local", local.state_fingerprint.clone()),
                ("rank", partition.rank.to_string()),
                ("ranks", partition.ranks.to_string()),
            ],
        );
        Ok(local)
    }
}

/// Retains physical parameter identities and arithmetic while narrowing the expert axis.
pub(crate) fn local_expert_spec(
    global: &GroupedGatedProductSpec,
    groups: std::ops::Range<usize>,
) -> Result<GroupedGatedProductSpec, Error> {
    let layout = match global.layout() {
        GatedProductGroupLayout::Packed { gate_up, down } => GatedProductGroupLayout::Packed {
            gate_up: gate_up.clone(),
            down: down.clone(),
        },
        GatedProductGroupLayout::Independent(parameters) => GatedProductGroupLayout::Independent(
            parameters
                .get(groups.clone())
                .ok_or_else(|| invalid("expert range exceeds parameters"))?
                .to_vec(),
        ),
        _ => return Err(invalid("unsupported expert layout")),
    };
    GroupedGatedProductSpec::new(
        i32::try_from(groups.len()).map_err(invalid)?,
        global.input_dimensions(),
        global.intermediate_dimensions(),
        global.output_dimensions(),
        global.policy(),
        layout,
    )
    .map(|spec| spec.with_reduction(global.reduction()))
}

impl TargetSpec {
    /// Authors EP ownership from this spec's already selected TP geometry.
    /// Router IDs remain checkpoint-global; the shared provider translates them
    /// to owner-local expert slots and performs the existing exchange.
    pub fn expert_realization(
        &self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<crate::ExpertRealizationPlan<GroupedGatedProductSpec>, Error> {
        let global = usize::try_from(self.config.experts.count).map_err(invalid)?;
        let groups = eredu_core::balanced_contiguous_range(
            global,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(invalid)?;
        let owner = eredu_runtime::ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP)
            .map_err(invalid)?;
        let units = self
            .units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                let UnitSpec::Decoder { feed_forward, .. } = unit else {
                    return None;
                };
                Some((
                    ordinal,
                    &feed_forward.feed_forward.parameters.routed.experts,
                ))
            })
            .map(|(ordinal, spec)| {
                if spec.group_count() != self.config.experts.count {
                    return Err(invalid(
                        "expert realization requires global bank cardinality",
                    ));
                }
                Ok((
                    (owner.clone(), ordinal),
                    local_expert_spec(spec, groups.clone())?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        crate::ExpertRealizationPlan::balanced(global, topology, units).map_err(invalid)
    }
}

impl TargetSpec {
    /// Consumes target geometry to bind exactly the admitted expert owner.
    pub fn with_expert_realization(
        self,
        realization: &crate::ExpertRealizationPlan<GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        let global = usize::try_from(self.config.experts.count).map_err(invalid)?;
        let groups = eredu_core::balanced_contiguous_range(
            global,
            realization.expert_parallel_size(),
            realization.expert_parallel_rank(),
            false,
        )
        .map_err(invalid)?;
        if realization.global_expert_count() != global
            || realization.local_global_group_indices() != groups.clone().collect::<Vec<_>>()
        {
            return Err(invalid(
                "expert realization ownership differs from target geometry",
            ));
        }
        let mut local = self;
        let mut count = 0;
        for (ordinal, unit) in local.units.iter_mut().enumerate() {
            let UnitSpec::Decoder { feed_forward, .. } = unit else {
                continue;
            };
            let experts = &feed_forward.feed_forward.parameters.routed.experts;
            if experts.group_count() as usize != global {
                return Err(invalid("expert realization is already bound"));
            }
            let expected = local_expert_spec(experts, groups.clone())?;
            let selected = realization
                .unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, ordinal)
                .ok_or_else(|| invalid("expert realization omits decoder unit"))?;
            if selected != &expected {
                return Err(invalid(
                    "expert realization differs from selected target parameters or equation",
                ));
            }
            feed_forward.feed_forward.parameters.routed.experts = selected.clone();
            feed_forward.validate()?;
            count += 1;
        }
        if count != realization.unit_specs().len() {
            return Err(invalid(
                "expert realization includes another execution unit",
            ));
        }
        Ok(local)
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TargetKernel<B> {
    /// Binds the selected local expert operators without changing global router
    /// identities, ordinary shared projections, state or checkpoint coordinates.
    /// Call during partition construction, before any execution units are built.
    pub fn with_expert_realization(
        mut self,
        realization: &crate::ExpertRealizationPlan<GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        let local = self.spec.clone().with_expert_realization(realization)?;
        let definition = if self.definition.parameters().partition_layout().is_some() {
            self.definition.clone()
        } else {
            self.parameter_root
                .project_owned_parameters(local.declare_parameters()?)?
        };
        self.spec = local;
        self.definition = definition;
        Ok(self)
    }
}
