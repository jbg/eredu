//! Invocation-scoped declarations authored from the retained target/prediction specs.
use super::{SelectedTargetExecution, UnitSpec};
use crate::{qwen4_exp::target::MixerSpec, speculative_execution::SpeculativeActivationExecution};
use eredu_core::{capture::*, intervention::*, speculative::*, *};

/// Loaded source/session authority and native collector facts. The composition
/// owner supplies the content identity of the retained prepared source graph;
/// discovery never reopens checkpoint files or hashes model payloads itself.
pub struct PredictionObservationBinding<'a> {
    /// Content-exact identity resolved by the source owner.
    pub artifact: artifact::ArtifactIdentity,
    /// Actual loaded session identity, also used by intervention admission.
    pub session: &'a str,
    /// Effective parameter overlay identity, if weights have been edited.
    pub overlay: Option<&'a str>,
    /// Native host conversion facts, independent of architecture hook coverage.
    pub observations: ObservationMechanisms,
    /// Native bounded transformations and physical-memory enforcement facts.
    pub captures: CaptureCapabilities,
    /// Native activation-edit arithmetic facts.
    pub interventions: InterventionMechanisms,
}

/// Joined logical ownership and exact loaded speculative capture admission.
/// The graph describes invocations and sharing, not native residency or memory usage.
#[derive(Debug, Clone)]
pub struct PredictionDiscovery {
    /// Architecture-authored operations, parameter groups, state edges and catalog.
    pub architecture: ArchitectureDescriptor,
    /// Support and loaded authority over that same catalog and node namespace.
    pub activations: SpeculativeActivationDiscovery,
}

impl SelectedTargetExecution {
    /// Projects the actual sequential target/MTP hook contract. Scope and tensor
    /// geometry come from retained architecture specifications, never path parsing
    /// in a backend. Observation and edit owners resolve through the logical graph;
    /// component score/readout decomposition and media processing remain partial.
    pub fn speculative_discovery(
        &self,
        execution: &SpeculativeActivationExecution,
        binding: PredictionObservationBinding<'_>,
    ) -> Result<PredictionDiscovery, CaptureError> {
        let prediction = self
            .prediction_spec()
            .map_err(|e| CaptureError::Invalid(e.to_string()))?;
        if binding.session.is_empty()
            || execution.depth != prediction.units.len()
            || execution.strategy != eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential
        {
            return Err(CaptureError::Invalid(
                "prediction observation binding differs from selected execution".into(),
            ));
        }
        let target = &self.plan.target.spec;
        let mut declaration = Declaration::new(target.boundary.geometry);
        let scope = SpeculativeCaptureScope::Target;
        declaration.owner = "target.embedding".into();
        declaration.point("readout.embedding", true, scope, false, None);
        for (ordinal, unit) in target.units.iter().enumerate() {
            let owner = format!("target.units.{ordinal}");
            declaration.owner = owner.clone();
            for seam in [UnitObservation::Input, UnitObservation::Output] {
                declaration.point(&seam.path(&unit.path()), true, scope, false, None);
            }
            let path = format!("model.layers.{}", unit.layer());
            match unit {
                UnitSpec::Lexical { .. } => {
                    for suffix in ["input", "write", "residual"] {
                        declaration.point(
                            &format!("{path}.lexical.{suffix}"),
                            true,
                            scope,
                            false,
                            None,
                        );
                    }
                }
                UnitSpec::Decoder {
                    mixer,
                    feed_forward,
                    ..
                } => {
                    declaration.decoder(&path, &owner, mixer, scope);
                    declaration.feed_forward(
                        &format!("{path}.mlp"),
                        &format!("{owner}.feed_forward"),
                        &feed_forward.feed_forward,
                        scope,
                    )?;
                }
            }
        }
        declaration.readout("readout", "target", scope, target.config.vocabulary);
        declaration.owner = "target.logits".into();
        declaration.publication(
            MODEL_LOGITS_OBSERVATION_PATH,
            scope,
            target.config.vocabulary,
        );
        let prepared =
            self.plan.prediction.as_ref().ok_or_else(|| {
                CaptureError::Invalid("missing selected prediction source".into())
            })?;
        for unit in &prepared.0.spec.units {
            let scope = SpeculativeCaptureScope::Prediction { depth: unit.depth };
            execution.validate_scope(scope)?;
            declaration.prediction(unit, target.config.vocabulary);
            declaration.feed_forward(
                &format!("mtp.layers.{}.mlp", unit.depth),
                &format!("prediction.{}.feed_forward", unit.depth),
                &unit.feed_forward.feed_forward,
                scope,
            )?;
        }
        declaration.points.sort_by(|a, b| a.path.cmp(&b.path));
        let catalog = ObservationCatalog {
            schema_version: DISCOVERY_SCHEMA_VERSION,
            points: declaration.points,
            completeness: DescriptionCompleteness::Partial(vec![
                "Execution and observation ownership are joined; component score/readout decomposition and media processing are not yet projected".into(),
            ]),
        };
        let (architecture, bindings) = super::graph::Graph::new(target, &prediction).join(
            catalog.clone(),
            &declaration.edits,
            execution,
        )?;
        for binding in &bindings {
            if declaration.bindings.get(&binding.node_id) != Some(&binding.scope) {
                return Err(CaptureError::Invalid(
                    "observation scope differs from logical graph ownership".into(),
                ));
            }
        }
        // The sealed executor discharges only invocation presence. Keep the
        // declaration conditional for other callers, just as prepared discovery
        // does for the existing families.
        let mut scoped_catalog = catalog.clone();
        for point in &mut scoped_catalog.points {
            point
                .requirements
                .retain(|r| *r != ObservationRequirement::PredictionExecution);
        }
        let mut support = eredu_runtime::inspection::observation_support(
            &scoped_catalog,
            eredu_runtime::inspection::ObservationExecutionContext {
                activation_inspection: self.selected.text().session().activation_inspection(),
                prediction_inspection: true,
                partitioned: false,
                selected: true,
                mechanisms: binding.observations,
            },
        );
        support.capture = binding.captures;
        let captures = CaptureDiscovery {
            artifact_identity: binding.artifact.to_string(),
            catalog,
            support,
        };
        let mut interventions = eredu_runtime::inspection::intervention_support(
            declaration.edits,
            &captures,
            &binding.interventions,
        );
        interventions.session_identity = Some(binding.session.into());
        let execution_identity = cache::derive_prompt_cache_architecture_fingerprint(
            "qwen4_exp.speculative.observations.v1",
            [
                ("target", format!("{:?}", self.selected)),
                ("prediction", format!("{prediction:?}")),
                ("state", format!("{:?}", self.prediction_state)),
                ("overlay", format!("{:?}", binding.overlay)),
            ],
        );
        Ok(PredictionDiscovery {
            architecture,
            activations: SpeculativeActivationDiscovery {
                schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                execution_identity,
                captures,
                interventions,
                bindings,
            },
        })
    }
}
/// Prediction hooks shared by low-level and ordinary retained discovery. Routed
/// feed-forward component decomposition remains a separate low-level declaration.
pub(super) fn prediction_points(
    target: &super::TargetSpec,
    prediction: &crate::qwen4_exp::mtp::PredictionSpec,
) -> Vec<ObservationPoint> {
    let mut declaration = Declaration::new(target.boundary.geometry);
    for unit in &prediction.units {
        declaration.prediction(unit, target.config.vocabulary);
    }
    declaration.points.sort_by(|a, b| a.path.cmp(&b.path));
    declaration.points
}

struct Declaration {
    geometry: eredu_nn::residual_streams::ResidualStreamGeometry,
    points: Vec<ObservationPoint>,
    edits: Vec<InterventionPoint>,
    owner: String,
    bindings: std::collections::BTreeMap<String, SpeculativeCaptureScope>,
}
impl Declaration {
    fn new(geometry: eredu_nn::residual_streams::ResidualStreamGeometry) -> Self {
        Self {
            geometry,
            points: vec![],
            edits: vec![],
            owner: String::new(),
            bindings: std::collections::BTreeMap::new(),
        }
    }
    fn prediction(&mut self, unit: &crate::qwen4_exp::mtp::PredictionUnitSpec, vocabulary: i32) {
        let scope = SpeculativeCaptureScope::Prediction { depth: unit.depth };
        let path = format!("mtp.layers.{}.prediction", unit.depth);
        let owner = format!("prediction.{}", unit.depth);
        self.owner = format!("{owner}.fusion");
        self.point(&format!("{path}.input"), true, scope, false, None);
        self.owner = format!("{owner}.embedding");
        self.point(&format!("{path}.embedding"), false, scope, false, None);
        self.owner = format!("{owner}.logits");
        self.publication(&format!("{path}.logits"), scope, vocabulary);
        self.owner = format!("{owner}.fusion");
        self.point(&format!("{path}.fusion"), true, scope, false, None);
        self.decoder(&path, &owner, &unit.mixer, scope);

        self.owner = format!("{owner}.feed_forward.residual");
        self.point(&format!("{path}.capture"), true, scope, false, None);
        // Collapse runs at the prediction root; only vocabulary projection
        // enters the nested readout scope.
        self.owner = format!("{owner}.readout");
        self.point(&format!("{path}.residual"), true, scope, false, None);
        self.point(&format!("{path}.normalized"), false, scope, false, None);
        self.owner = format!("{owner}.head");
        self.projection(&format!("{path}.readout"), scope, vocabulary);
    }
    fn point(
        &mut self,
        path: &str,
        streams: bool,
        scope: SpeculativeCaptureScope,
        readonly: bool,
        vocabulary: Option<i32>,
    ) {
        use SymbolicDimension as D;
        let mut axes = vec![
            TensorAxis {
                name: "batch".into(),
                dimension: D::Batch,
            },
            TensorAxis {
                name: "sequence".into(),
                dimension: D::Sequence,
            },
        ];
        if streams {
            axes.push(TensorAxis {
                name: "stream".into(),
                dimension: D::Known(self.geometry.streams() as usize),
            });
        }
        axes.push(TensorAxis {
            name: if vocabulary.is_some() {
                "vocabulary"
            } else {
                "hidden"
            }
            .into(),
            dimension: D::Known(vocabulary.unwrap_or(self.geometry.hidden_size()) as usize),
        });
        let meaning = if streams {
            "Complete uncollapsed residual streams"
        } else if vocabulary.is_some() {
            "Vocabulary projection"
        } else {
            "Collapsed hidden-width activation"
        };
        self.tensor_point(
            path,
            axes,
            ObservationDtype::Floating,
            scope,
            readonly,
            meaning,
        );
    }
    fn tensor_point(
        &mut self,
        path: &str,
        axes: Vec<TensorAxis>,
        dtype: ObservationDtype,
        scope: SpeculativeCaptureScope,
        readonly: bool,
        meaning: &str,
    ) {
        let node = self.owner.clone();
        self.bindings.insert(node.clone(), scope);
        if !readonly {
            self.edits.push(InterventionPoint {
                path: path.into(),
                node_id: node.clone(),
                stage: InterventionStage::Activation,
                axes: axes.clone(),
                dtypes: vec![
                    InterventionDtype::Float32,
                    InterventionDtype::Float16,
                    InterventionDtype::Bfloat16,
                ],
                operations: vec![
                    InterventionKind::Zero,
                    InterventionKind::Scale,
                    InterventionKind::Mask,
                    InterventionKind::MaskComponents,
                    InterventionKind::Replace,
                    InterventionKind::Add,
                ],
                score_stages: vec![],
                prefill: ObservationSupportStatus::Unverified("collector not bound".into()),
                decode: ObservationSupportStatus::Unverified("collector not bound".into()),
                conditions: vec![],
                routing: None,
                routed_units: None,
            });
        }
        let mut requirements = vec![ObservationRequirement::ActivationHooks];
        if scope != SpeculativeCaptureScope::Target {
            requirements.push(ObservationRequirement::PredictionExecution);
        }
        self.points.push(ObservationPoint {
            path: path.into(),
            node_id: node,
            meaning: meaning.into(),
            value_type: ObservationValueType::Tensor,
            dtype,
            axes: Some(axes),
            prefill: true,
            decode: true,
            requirements,
            position: if readonly {
                ObservationPosition::ReadOnly
            } else {
                ObservationPosition::BeforeIntervention
            },
            retained_bytes: None,
            host_bytes: None,
        });
    }
    fn decoder(
        &mut self,
        path: &str,
        owner: &str,
        mixer: &MixerSpec,
        scope: SpeculativeCaptureScope,
    ) {
        self.owner = format!("{owner}.mixer");
        match mixer {
            MixerSpec::Recurrent(spec) => {
                let spec = &spec.mixer;
                for (suffix, axis, width, readonly) in [
                    ("qkv.projected", "qkv_channel", spec.input_qkv.output, true),
                    ("qkv.convolved", "qkv_channel", spec.input_qkv.output, true),
                    (
                        "update.projected",
                        "value_head",
                        spec.input_beta.output,
                        true,
                    ),
                    (
                        "decay.projected",
                        "value_head",
                        spec.input_decay.output,
                        true,
                    ),
                    (
                        "gate.projected",
                        "value_channel",
                        spec.input_gate.output,
                        true,
                    ),
                    ("channels", "value_channel", spec.output.input, false),
                    ("write_input", "value_channel", spec.output.input, true),
                ] {
                    self.channels(
                        &format!("{path}.mixer.{suffix}"),
                        axis,
                        width,
                        scope,
                        readonly,
                    );
                }
            }
            MixerSpec::Indexed(spec) => {
                for (suffix, readonly) in [("channels", false), ("write_input", true)] {
                    self.channels(
                        &format!("{path}.attention.{suffix}"),
                        "attention_channel",
                        spec.projections[3].input,
                        scope,
                        readonly,
                    );
                }
                // Validated QSA geometry includes the complete-block budget plus
                // its incomplete tail, even when most slots are padding.
                let selection = spec.indexer.selection;
                self.tensor_point(
                    &format!("{path}.attention.selected_positions"),
                    Self::sequence_axes(
                        "selected_position",
                        selection.token_budget + selection.ratio - 1,
                    ),
                    ObservationDtype::Integer,
                    scope,
                    true,
                    "Original causal K/V positions; unused slots contain -1",
                );
            }
        }
        let mixer = match mixer {
            MixerSpec::Recurrent(_) => "mixer",
            MixerSpec::Indexed(_) => "attention",
        };
        for part in [mixer, "feed_forward"] {
            self.owner = format!(
                "{owner}.{}.residual",
                if part == "feed_forward" {
                    "feed_forward"
                } else {
                    "mixer"
                }
            );
            for suffix in ["input", "write", "output", "residual"] {
                self.point(
                    &format!("{path}.{part}.{suffix}"),
                    suffix == "residual",
                    scope,
                    false,
                    None,
                );
            }
        }
    }
    fn sequence_axes(axis: &str, width: i32) -> Vec<TensorAxis> {
        vec![
            TensorAxis {
                name: "batch".into(),
                dimension: SymbolicDimension::Batch,
            },
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: axis.into(),
                dimension: SymbolicDimension::Known(width as usize),
            },
        ]
    }
    fn channels(
        &mut self,
        path: &str,
        axis: &str,
        width: i32,
        scope: SpeculativeCaptureScope,
        readonly: bool,
    ) {
        self.tensor_point(
            path,
            Self::sequence_axes(axis, width),
            ObservationDtype::Floating,
            scope,
            readonly,
            "Internal projection channels in operator storage order",
        );
    }
    fn feed_forward(
        &mut self,
        path: &str,
        owner: &str,
        spec: &crate::shared_routed::SharedRoutedGatedProductSpec,
        scope: SpeculativeCaptureScope,
    ) -> Result<(), CaptureError> {
        let node = format!("{owner}.router");
        let edit = crate::discovery::routing_intervention_point(
            &node,
            path,
            spec.router.selection(),
            1,
            spec.router.coefficient_scale().is_some(),
        )
        .ok_or_else(|| {
            CaptureError::Unsupported("routing policy has no intervention declaration".into())
        })?;
        self.bindings.insert(node, scope);
        self.edits.push(edit);
        self.owner = format!("{owner}.routed");
        self.routed_units(path, spec, scope)?;
        self.owner = format!("{owner}.shared");
        let shared = format!("{path}.shared_expert");
        for (suffix, axis, width, readonly) in [
            (
                "feed_forward.units",
                "shared_unit",
                spec.shared[0].output,
                false,
            ),
            (
                "feed_forward.write_input",
                "shared_unit",
                spec.shared[2].input,
                true,
            ),
            ("feed_forward.write", "hidden", spec.shared[2].output, false),
            (
                "feed_forward.output",
                "hidden",
                spec.shared[2].output,
                false,
            ),
            ("gate.input", "hidden", spec.shared_gate.input, true),
            (
                "gate.projection_input",
                "hidden",
                spec.shared_gate.input,
                true,
            ),
            ("gate", "gate", spec.shared_gate.output, false),
        ] {
            self.channels(&format!("{shared}.{suffix}"), axis, width, scope, readonly);
        }
        self.owner = format!("{owner}.sum");
        self.channels(
            &format!("{path}.output"),
            "hidden",
            spec.shared[2].output,
            scope,
            false,
        );
        use RoutingObservationField as F;
        for field in [
            F::SelectedExperts,
            F::SelectedScores,
            F::Coefficients,
            F::RoutedOutput,
            F::SharedOutput,
            F::CombinedOutput,
        ] {
            self.owner = format!(
                "{owner}.{}",
                match field {
                    F::SelectedExperts | F::SelectedScores | F::Coefficients => "router",
                    F::RoutedOutput => "routed",
                    F::SharedOutput => "shared",
                    _ => "sum",
                }
            );
            let axes = match field {
                F::SelectedExperts | F::SelectedScores | F::Coefficients => vec![
                    TensorAxis {
                        name: "token".into(),
                        dimension: SymbolicDimension::TokenRows,
                    },
                    TensorAxis {
                        name: "selected_expert".into(),
                        dimension: SymbolicDimension::Known(
                            spec.router.selection().top_k() as usize
                        ),
                    },
                ],
                _ => Self::sequence_axes("hidden", spec.shared[2].output),
            };
            self.tensor_point(
                &field.path(path),
                axes,
                if field == F::SelectedExperts {
                    ObservationDtype::Integer
                } else {
                    ObservationDtype::Floating
                },
                scope,
                true,
                "Normalized routed-expert result after dispatch",
            );
            self.points
                .last_mut()
                .expect("declared event")
                .requirements
                .push(ObservationRequirement::RoutingEvents);
        }
        Ok(())
    }
    fn routed_units(
        &mut self,
        routing: &str,
        spec: &crate::shared_routed::SharedRoutedGatedProductSpec,
        scope: SpeculativeCaptureScope,
    ) -> Result<(), CaptureError> {
        let geometry = RoutedUnitGeometry {
            experts: spec.experts.group_count() as u64,
            units_per_expert: spec.experts.intermediate_dimensions() as u64,
            routes_per_token: spec.router.selection().top_k() as u64,
        };
        // Components name logical expert units; the native source contains only
        // selected routes. This virtual extent must never allocate expert-dense storage.
        let components =
            usize::try_from(geometry.components()?).map_err(|_| CaptureError::Overflow)?;
        for (suffix, position) in [
            ("units", ObservationPosition::BeforeIntervention),
            ("units.effective", ObservationPosition::AfterIntervention),
        ] {
            let path = format!("{routing}.{suffix}");
            self.tensor_point(
                &path,
                vec![
                    TensorAxis { name: "token".into(), dimension: SymbolicDimension::TokenRows },
                    TensorAxis { name: "route".into(), dimension: SymbolicDimension::Known(geometry.routes_per_token as usize) },
                    TensorAxis { name: "component".into(), dimension: SymbolicDimension::Known(geometry.units_per_expert as usize) },
                ],
                ObservationDtype::Floating,
                scope,
                true,
                "Selected expert units before down projection and route weighting, with original route identity",
            );
            let point = self.points.last_mut().expect("declared sparse point");
            point.position = position;
            point.value_type = ObservationValueType::RoutedUnits {
                routing: routing.into(),
                geometry,
            };
            if position == ObservationPosition::BeforeIntervention {
                self.edits.push(InterventionPoint {
                    path,
                    node_id: point.node_id.clone(),
                    stage: InterventionStage::Activation,
                    axes: vec![
                        TensorAxis { name: "token".into(), dimension: SymbolicDimension::TokenRows },
                        TensorAxis { name: "component".into(), dimension: SymbolicDimension::Known(components) },
                    ],
                    dtypes: vec![InterventionDtype::Float32, InterventionDtype::Float16, InterventionDtype::Bfloat16],
                    operations: vec![InterventionKind::Zero, InterventionKind::Scale, InterventionKind::Mask, InterventionKind::MaskComponents, InterventionKind::Replace, InterventionKind::Add],
                    score_stages: vec![],
                    prefill: ObservationSupportStatus::Unverified("collector not bound".into()),
                    decode: ObservationSupportStatus::Unverified("collector not bound".into()),
                    conditions: vec!["Global component = expert * units_per_expert + unit; edits follow selected routes before down projection and weighting".into()],
                    routing: None,
                    routed_units: Some(RoutedUnitInterventionPoint { routing: routing.into(), geometry }),
                });
            }
        }
        Ok(())
    }
    fn publication(&mut self, path: &str, scope: SpeculativeCaptureScope, vocabulary: i32) {
        self.point(path, false, scope, false, Some(vocabulary));
        let edit = self.edits.last_mut().expect("declared publication hook");
        edit.stage = InterventionStage::LogitsBeforeSampling;
        edit.operations.push(InterventionKind::MaskLogits);
        self.points
            .last_mut()
            .expect("declared publication point")
            .meaning =
            "Authoritative vocabulary logits before sampling and logit processing".into();
    }
    fn projection(&mut self, path: &str, scope: SpeculativeCaptureScope, vocabulary: i32) {
        self.point(
            &format!("{path}.projection_input"),
            false,
            scope,
            true,
            None,
        );
        self.point(
            &format!("{path}.linear"),
            false,
            scope,
            false,
            Some(vocabulary),
        );
    }
    fn readout(
        &mut self,
        path: &str,
        owner: &str,
        scope: SpeculativeCaptureScope,
        vocabulary: i32,
    ) {
        self.owner = format!("{owner}.readout");
        self.point(&format!("{path}.residual"), true, scope, false, None);
        self.point(&format!("{path}.normalized"), false, scope, false, None);
        self.owner = format!("{owner}.head");
        self.projection(path, scope, vocabulary);
    }
}
