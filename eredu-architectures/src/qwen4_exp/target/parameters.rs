//! Canonical target ownership and physical geometry without native modules.
use super::*;
use eredu_nn::{
    CausalDepthwiseConvolutionSpec, GatedProductGroupLayout, GroupedProjectionSpec,
    NormalizationConstructionSpec, NormalizationScale, ParameterMetadata,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, MemberSharding,
    OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec,
    ParameterRole,
};

#[derive(Default)]
struct Declaration {
    members: Vec<ParameterMemberSpec>,
    formats: BTreeMap<String, LinearFormatSpec>,
}
impl Declaration {
    fn parameter(&mut self, spec: &ParameterSpec, shape: &[i32]) {
        self.members.push(
            ParameterMemberSpec::new(
                spec.id.as_str(),
                shape.iter().map(|&n| n as usize).collect::<Vec<_>>(),
                MemberSharding::Replicated,
            )
            .with_parameter_metadata(&ParameterMetadata::from_spec(spec, spec.trainable)),
        );
    }

    fn projection(
        &mut self,
        weight: &ParameterSpec,
        bias: Option<&ParameterSpec>,
        format: &LinearFormatSpec,
        shape: &[i32],
    ) {
        self.parameter(weight, shape);
        self.formats
            .insert(weight.id.as_str().into(), format.clone());
        if let Some(bias) = bias {
            self.parameter(bias, &shape[..shape.len() - 1]);
        }
    }

    fn linear(&mut self, spec: &LinearSpec) {
        self.projection(
            &spec.weight,
            spec.bias.as_ref(),
            &spec.format,
            &[spec.output, spec.input],
        );
    }

    fn norm(&mut self, spec: &NormalizationConstructionSpec) {
        match &spec.scale {
            NormalizationScale::Learned(weight)
            | NormalizationScale::LearnedOffset { weight, .. } => {
                self.parameter(weight, &[spec.dimensions]);
            }
            NormalizationScale::Unit => {}
        }
    }

    fn residual(&mut self, spec: &eredu_nn::residual_streams::GatedResidualSpec) {
        self.norm(&spec.normalization);
        self.linear(&spec.down);
        self.linear(&spec.up);
        if let Some(injection) = &spec.injection {
            self.linear(injection);
        }
    }

    fn convolution(&mut self, spec: &CausalDepthwiseConvolutionSpec) {
        self.parameter(&spec.weight, &[spec.channels, 1, spec.kernel_size]);
        if let Some(bias) = &spec.bias {
            self.parameter(bias, &[spec.channels]);
        }
    }

    fn grouped(&mut self, spec: &GroupedProjectionSpec, shape: &[i32]) {
        self.projection(spec.weight(), spec.bias(), spec.format(), shape);
    }

    fn feed_forward(&mut self, spec: &FeedForwardSublayerSpec) -> Result<(), Error> {
        spec.validate()?;
        self.residual(&spec.residual);
        let ff = &spec.feed_forward;
        let router = eredu_runtime::top_k_group_selector_logical_group("router", &ff.router)
            .map_err(Error::backend_source)?;
        self.members.extend(router.members().iter().cloned());
        self.formats.insert(
            ff.router.weight().id.as_str().into(),
            ff.router.format().clone(),
        );
        let e = &ff.experts;
        let intermediate = e.intermediate_dimensions();
        match e.layout() {
            GatedProductGroupLayout::Packed { gate_up, down } => {
                self.grouped(
                    gate_up,
                    &[e.group_count(), 2 * intermediate, e.input_dimensions()],
                );
                self.grouped(
                    down,
                    &[e.group_count(), e.output_dimensions(), intermediate],
                );
            }
            GatedProductGroupLayout::Independent(experts) => {
                for expert in experts {
                    self.grouped(expert.gate(), &[intermediate, e.input_dimensions()]);
                    self.grouped(expert.up(), &[intermediate, e.input_dimensions()]);
                    self.grouped(expert.down(), &[e.output_dimensions(), intermediate]);
                }
            }
            _ => return Err(Error::backend("target declares an unknown expert layout")),
        }
        for projection in &ff.shared {
            self.linear(projection);
        }
        self.linear(&ff.shared_gate);
        Ok(())
    }

    fn unit(&mut self, spec: &UnitSpec) -> Result<(), Error> {
        match spec {
            UnitSpec::Lexical { spec, .. } => {
                spec.validate()?;
                // Table rows and their scalar companions belong to the row provider.
                self.linear(&spec.key);
                self.linear(&spec.value);
                self.norm(&spec.key_norm);
                self.norm(&spec.query_norm);
                self.norm(&spec.convolution_norm);
                self.convolution(&spec.convolution);
            }
            UnitSpec::Decoder {
                mixer,
                feed_forward,
                ..
            } => {
                match mixer {
                    MixerSpec::Recurrent(spec) => {
                        spec.validate()?;
                        self.residual(&spec.residual);
                        let m = &spec.mixer;
                        for p in [&m.input_qkv, &m.input_gate, &m.input_beta, &m.input_decay] {
                            self.linear(p);
                        }
                        self.convolution(&m.convolution);
                        self.parameter(&m.decay_bias, &[m.value_heads]);
                        self.parameter(&m.transition_log, &[m.value_heads]);
                        self.parameter(&m.normalization_weight, &[m.value_head_dim]);
                        self.linear(&m.output);
                    }
                    MixerSpec::Indexed(spec) => {
                        spec.validate().map_err(Error::backend_source)?;
                        self.residual(&spec.residual);
                        self.linear(&spec.indexer.query_key);
                        self.norm(&spec.indexer.query_norm);
                        self.norm(&spec.indexer.key_norm);
                        for p in &spec.projections {
                            self.linear(p);
                        }
                        self.norm(&spec.query_norm);
                        self.norm(&spec.key_norm);
                    }
                }
                self.feed_forward(feed_forward)?;
            }
        }
        Ok(())
    }

    fn group(&mut self, name: &str, role: ParameterRole) -> Result<ParameterGroupSpec, Error> {
        if self.members.iter().any(|member| {
            member
                .global_shape()
                .iter()
                .any(|&n| n == 0 || n > i32::MAX as usize)
        }) {
            return Err(Error::backend(format!(
                "target parameter group {name} has invalid native dimensions"
            )));
        }
        let group = ParameterGroupSpec::new(name, role, std::mem::take(&mut self.members))
            .map_err(Error::backend_source)?;
        eredu_runtime::expand_linear_format_parameter_group(group, |member| {
            Ok(self.formats.get(member.target()).cloned())
        })
        .map_err(Error::backend_source)
    }
}

impl TargetSpec {
    /// Projects canonical storage from the validated target declaration.
    pub fn parameter_description(&self) -> Result<ArchitectureParameterDescription, Error> {
        Ok(self.definition()?.parameters().clone())
    }

    /// Validates target invocations, storage, state and transport without native resources.
    pub fn definition(&self) -> Result<eredu_runtime::ComponentDefinition, Error> {
        self.definition_with_parameters(self.declare_parameters()?)
    }

    fn definition_with_parameters(
        &self,
        parameters: ArchitectureParameterDescription,
    ) -> Result<eredu_runtime::ComponentDefinition, Error> {
        eredu_runtime::ComponentDefinition::new(
            "qwen4_exp_target",
            parameters,
            self.units
                .iter()
                .enumerate()
                .map(|(index, unit)| eredu_runtime::ComponentUnit {
                    path: unit.path(),
                    state: index..index + 1,
                })
                .collect(),
            vec![crate::transport::decoder()],
            crate::decoder::TARGET_EXECUTION_GROUP,
            vec![],
            self.state_layout()?,
            eredu_runtime::ArchitectureStatePartitionPlan::new([
                eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..self.units.len()),
            ]),
        )
        .map_err(Error::from)
    }

    pub(super) fn declare_parameters(&self) -> Result<ArchitectureParameterDescription, Error> {
        self.limits.validate_for_config(&self.config)?;
        self.state_layout()?;
        let graph = ExecutionGraph::chain([crate::decoder::TARGET_EXECUTION_GROUP])
            .map_err(Error::backend_source)?;
        let layout =
            ExecutionUnitLayout::new(&graph, [self.units.len()]).map_err(Error::backend_source)?;
        let mut declaration = Declaration::default();
        declaration.projection(
            &self.embedding.weight,
            None,
            &self.embedding.format,
            &[self.embedding.vocabulary, self.embedding.dimensions],
        );
        let mut groups = vec![OwnedParameterGroupSpec::new(
            if self.head.is_none() {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role("embedding")
            },
            declaration.group("embedding", ParameterRole::Vocabulary)?,
        )];
        declaration.residual(&self.boundary);
        groups.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("norm"),
            declaration.group("readout", ParameterRole::Replicated)?,
        ));
        if let Some(head) = &self.head {
            declaration.linear(head);
            groups.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("output"),
                declaration.group("output", ParameterRole::Vocabulary)?,
            ));
        }
        for (index, unit) in self.units.iter().enumerate() {
            if let UnitSpec::Decoder { feed_forward, .. } = unit {
                if feed_forward.feed_forward.layer != index {
                    return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                        component: "qwen4_exp_target".into(),
                        resource: format!("expert owner for {}", unit.path()),
                    }
                    .into());
                }
            }
            declaration.unit(unit)?;
            groups.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout.group_id(0).expect("target group").clone(),
                    index,
                ),
                declaration.group(&unit.path(), ParameterRole::Replicated)?,
            ));
        }
        ArchitectureParameterDescription::from_owned(&graph, &layout, groups)
            .map_err(Error::backend_source)
    }
}
