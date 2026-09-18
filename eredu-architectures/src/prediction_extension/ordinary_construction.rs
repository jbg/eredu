//! Cold prediction construction with one frame per selected family.
use super::*;

pub(super) struct Construction<'a, B: eredu_nn::NeuralBackend> {
    pub(super) extension: &'a PredictionExtensionPlan,
    pub(super) topology: ParallelRankTopology,
    pub(super) tensor_rank: ParallelRankTopology,
    pub(super) tasks: &'a [ReplicatedTextMaterializationTask],
    pub(super) source_context: &'a <B::Tensor as Tensor>::Context,
    pub(super) execution_context: &'a <B::Tensor as Tensor>::Context,
}
impl<B> Construction<'_, B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    // Keep cold family constructors in separate frames; none needs another family's locals.
    #[inline(never)]
    pub(super) fn v3(
        self,
        args: &crate::deepseek::V3Args,
    ) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError> {
        let Self {
            extension,
            topology,
            tensor_rank,
            tasks,
            source_context,
            execution_context,
        } = self;
        let mut formats = args.linear_formats.clone();
        formats.extend(
            tasks
                .iter()
                .filter(|task| {
                    matches!(
                        task.role(),
                        eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                    )
                })
                .map(|task| (task.name().to_owned(), task.executable())),
        );
        let target_args =
            crate::deepseek::v3_with_checkpoint_formats(args, formats).map_err(invalid)?;
        let parameters = crate::deepseek::parallel::v3_parameter_description(&target_args)
            .map_err(|error| invalid(error.to_string()))?;
        let layout =
            crate::partitioned_execution::derive_partitioned_local_layout(&parameters, tensor_rank)
                .map_err(invalid)?;
        let geometry = crate::deepseek::parallel::v3_local_geometry(&target_args, &layout)
            .map_err(|error| invalid(error.to_string()))?;
        // Retain the selected local policy before its geometry moves into the
        // executable model. Empty current caches cannot supply these widths.
        let target = usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("DeepSeek-V3 target count exceeds usize"))?;
        let mut state = Vec::with_capacity(extension.depth());
        for depth in 0..extension.depth() {
            let ordinal = target
                .checked_add(depth)
                .ok_or_else(|| invalid("DeepSeek-V3 prediction state ordinal overflowed"))?;
            let policy = geometry.state_layout().layer(ordinal).ok_or_else(|| {
                invalid("DeepSeek-V3 prediction depth has no selected state policy")
            })?;
            // The authoritative V3 state producer is compressed-only. Keep
            // this fixed, allocation-free clone explicit at the source seam.
            if !matches!(policy, LayerCachePolicy::CompressedLatentRotary { .. }) {
                return Err(invalid("DeepSeek-V3 prediction state is not compressed"));
            }
            state.push((ordinal, policy.clone()));
        }
        let source_layout = if tasks.iter().any(|task| {
            matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            )
        }) {
            let source_parameters = crate::deepseek::parallel::v3_parameter_description(args)
                .map_err(|error| invalid(error.to_string()))?;
            Some(std::sync::Arc::new(
                crate::partitioned_execution::derive_partitioned_transform_source_layout(
                    &source_parameters,
                    &parameters,
                    tensor_rank,
                )
                .map_err(invalid)?,
            ))
        } else {
            None
        };
        let source = match &source_layout {
            Some(layout) => crate::deepseek::v3::Model::<B>::new_parallel(
                args.clone(),
                crate::deepseek::parallel::v3_local_geometry(args, layout)
                    .map_err(|error| invalid(error.to_string()))?,
                source_context,
            ),
            None => crate::deepseek::v3::Model::<B>::new(args.clone(), source_context),
        }
        .map_err(|error| invalid(error.to_string()))?;
        let local =
            crate::deepseek::v3::Model::<B>::new_parallel(target_args, geometry, execution_context)
                .map_err(|error| invalid(error.to_string()))?;
        let descriptor =
            crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                extension.complete_architecture().clone(),
            )
            .architecture_descriptor();
        let mut units = Vec::with_capacity(extension.depth());
        let mut coordinate_rows = Vec::with_capacity(extension.depth());
        let mut source_specs = Vec::with_capacity(extension.depth());
        let mut local_specs = Vec::with_capacity(extension.depth());
        for depth in 0..extension.depth() {
            let source_spec = crate::deepseek::mtp::V3PredictionLayerSpec::new(
                source.unit_construction_args(),
                depth,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let local_spec = crate::deepseek::mtp::V3PredictionLayerSpec::new(
                local.unit_construction_args(),
                depth,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let source_unit = crate::deepseek::v3::Unit::Prediction(
                source_spec
                    .instantiate::<B>(source_context)
                    .map_err(|error| invalid(error.to_string()))?,
            );
            let mut local_unit = crate::deepseek::v3::Unit::Prediction(
                local_spec
                    .instantiate::<B>(execution_context)
                    .map_err(|error| invalid(error.to_string()))?,
            );
            source_specs.push(source_spec);
            local_specs.push(local_spec);
            let mut coordinate_row = None;
            if let crate::deepseek::v3::Unit::Prediction(prediction) = &mut local_unit {
                if let crate::deepseek::block::V3FeedForward::Routed(moe) =
                    &mut prediction.decoder.feed_forward
                {
                    let scope = descriptor.component_scopes.iter()
                        .find(|scope| matches!(scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth: selected } if selected == depth))
                        .ok_or_else(|| invalid("prepared prediction has no component scope"))?;
                    let [component] = scope.routed_components.as_slice() else {
                        return Err(invalid(
                            "V3 prediction must declare one resident routed bank",
                        ));
                    };
                    let coordinates = crate::component_partition::derive_coordinates_for_experts(
                        component,
                        &layout,
                        &(0..component.expert_count).collect::<Vec<_>>(),
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    let coordinates = std::sync::Arc::new(coordinates.units().clone());
                    coordinate_row = Some(coordinates.clone());
                    moe.bind_shared_resident_unit_coordinates(
                        coordinates,
                        topology.topology().world_size() > 1,
                    );
                }
            }
            coordinate_rows.push(coordinate_row);
            units.push(
                PreparedPredictionUnit::new(source_unit, local_unit, tasks)?
                    .with_source_layout(source_layout.clone()),
            );
        }
        let construction = std::sync::Arc::new(construction::PreparedPredictionConstruction::new(
            source_specs,
            local_specs,
            tasks.to_vec(),
            units.iter().map(|unit| unit.tasks.clone()).collect(),
            source_layout,
            coordinate_rows,
            state.clone(),
            topology.topology().world_size() > 1,
        ));
        Ok(PreparedPredictionExtension::DeepSeekV3 {
            layout: std::sync::Arc::new(layout),
            parameters: std::sync::Arc::new(parameters),
            units,
            state,
            construction,
        })
    }
    // Keep cold family constructors in separate frames; none needs another family's locals.
    #[inline(never)]
    pub(super) fn v4(
        self,
        args: &crate::deepseek::V4Args,
    ) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError> {
        let Self {
            extension,
            topology,
            tensor_rank,
            tasks,
            source_context,
            execution_context,
        } = self;
        let mut formats = args.linear_formats.clone();
        formats.extend(
            tasks
                .iter()
                .filter(|task| {
                    matches!(
                        task.role(),
                        eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                    )
                })
                .map(|task| (task.name().to_owned(), task.executable())),
        );
        let target_args =
            crate::deepseek::v4_with_checkpoint_formats(args, formats).map_err(invalid)?;
        let parameters = crate::deepseek::parallel::v4_parameter_description(&target_args)
            .map_err(|error| invalid(error.to_string()))?;
        let layout =
            crate::partitioned_execution::derive_partitioned_local_layout(&parameters, tensor_rank)
                .map_err(invalid)?;
        let geometry = crate::deepseek::parallel::v4_local_geometry(&target_args, &layout)
            .map_err(|error| invalid(error.to_string()))?;
        let state_layout = crate::deepseek::v4::state_layout(geometry.args())
            .map_err(|error| invalid(error.to_string()))?;
        let source_layout = if tasks.iter().any(|task| {
            matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            )
        }) {
            let source_parameters = crate::deepseek::parallel::v4_parameter_description(args)
                .map_err(|error| invalid(error.to_string()))?;
            Some(std::sync::Arc::new(
                crate::partitioned_execution::derive_partitioned_transform_source_layout(
                    &source_parameters,
                    &parameters,
                    tensor_rank,
                )
                .map_err(invalid)?,
            ))
        } else {
            None
        };
        let source = match &source_layout {
            Some(layout) => crate::deepseek::v4::Model::<B>::new_parallel(
                args.clone(),
                crate::deepseek::parallel::v4_local_geometry(args, layout)
                    .map_err(|error| invalid(error.to_string()))?,
                source_context,
            ),
            None => crate::deepseek::v4::Model::<B>::new(args.clone(), source_context),
        }
        .map_err(|error| invalid(error.to_string()))?;
        let local =
            crate::deepseek::v4::Model::<B>::new_parallel(target_args, geometry, execution_context)
                .map_err(|error| invalid(error.to_string()))?;
        let target = usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("DeepSeek-V4 target count exceeds usize"))?;
        let descriptor =
            crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                extension.complete_architecture().clone(),
            )
            .architecture_descriptor();
        let mut source_specs = Vec::with_capacity(extension.depth());
        let mut local_specs = Vec::with_capacity(extension.depth());
        let mut coordinates = Vec::with_capacity(extension.depth());
        let mut units = Vec::with_capacity(extension.depth());
        let mut state = Vec::with_capacity(extension.depth());
        for depth in 0..extension.depth() {
            let ordinal = target + depth;
            let source_spec = source
                .prediction_unit_spec(depth)
                .map_err(|error| invalid(error.to_string()))?;
            let local_spec = local
                .prediction_unit_spec(depth)
                .map_err(|error| invalid(error.to_string()))?;
            let source_unit = source_spec
                .instantiate::<B>(source_context)
                .map_err(|error| invalid(error.to_string()))?;
            let mut local_unit = local_spec
                .instantiate::<B>(execution_context)
                .map_err(|error| invalid(error.to_string()))?;
            source_specs.push(source_spec);
            local_specs.push(local_spec);
            let policy = state_layout.layer(ordinal).cloned().ok_or_else(|| {
                invalid(format!(
                    "DeepSeek-V4 prediction depth {depth} has no state policy"
                ))
            })?;
            let routed = match &mut local_unit {
                crate::deepseek::v4::Unit::Prediction(prediction) => {
                    &mut prediction.decoder.feed_forward
                }
                crate::deepseek::v4::Unit::Dspark(block) => &mut block.feed_forward,
                crate::deepseek::v4::Unit::Target(_) => {
                    return Err(invalid("prediction construction returned a target unit"));
                }
            };
            {
                let scope = descriptor.component_scopes.iter().find(|scope| matches!(
                    scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth: selected } if selected == depth
                ) || matches!(scope.kind, eredu_core::component::ComponentExecutionScopeKind::FusedPrediction)).ok_or_else(|| invalid("prepared V4 prediction has no component scope"))?;
                let mut components = scope
                    .routed_components
                    .iter()
                    .filter(|component| component.layer_index == ordinal);
                let component = components.next().ok_or_else(|| {
                    invalid("V4 prediction must declare its resident routed bank")
                })?;
                if components.next().is_some() {
                    return Err(invalid(
                        "V4 prediction declares multiple resident banks for one block",
                    ));
                }
                let derived = crate::component_partition::derive_coordinates_for_experts(
                    component,
                    &layout,
                    &(0..component.expert_count).collect::<Vec<_>>(),
                )
                .map_err(|error| invalid(error.to_string()))?;
                let map = std::sync::Arc::new(derived.units().clone());
                routed.bind_shared_resident_unit_coordinates(
                    map.clone(),
                    topology.topology().world_size() > 1,
                );
                coordinates.push(Some(map));
            }
            units.push(
                PreparedPredictionUnit::new(source_unit, local_unit, tasks)?
                    .with_source_layout(source_layout.clone()),
            );
            state.push((ordinal, policy));
        }
        let rows = units.iter().map(|unit| unit.tasks.clone()).collect();
        if args.dspark.is_some() {
            let strategy = DsparkPredictionStrategy::from_args(args)?;
            let source_static_spec = source
                .dspark_static_spec()
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("source DSpark model has no fused pinned modules"))?;
            let local_static_spec = local
                .dspark_static_spec()
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("local DSpark model has no fused pinned modules"))?;
            // Initial load uses the same typed declaration worker as retained
            // reconstruction; no placeholder tensor is its own parameter source.
            let static_modules = PreparedPredictionUnit::new_shared(
                source_static_spec
                    .instantiate::<B>(source_context)
                    .map_err(|error| invalid(error.to_string()))?,
                local_static_spec
                    .instantiate::<B>(execution_context)
                    .map_err(|error| invalid(error.to_string()))?,
                tasks,
            )?
            .with_source_layout(source_layout.clone());
            let shared = construction::DsparkConstruction {
                source: source_static_spec,
                local: local_static_spec,
                strategy: strategy.clone(),
                tasks: static_modules.tasks.clone(),
            };
            let construction =
                std::sync::Arc::new(construction::PreparedPredictionConstruction::v4(
                    source_specs,
                    local_specs,
                    tasks.to_vec(),
                    rows,
                    source_layout,
                    coordinates,
                    state.clone(),
                    Some(shared),
                    topology.topology().world_size() > 1,
                ));
            Ok(PreparedPredictionExtension::DeepSeekV4Dspark {
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(parameters),
                extension: PreparedDsparkPredictionExtension {
                    strategy,
                    static_modules,
                },
                units,
                state,
                construction,
            })
        } else {
            let construction =
                std::sync::Arc::new(construction::PreparedPredictionConstruction::v4(
                    source_specs,
                    local_specs,
                    tasks.to_vec(),
                    rows,
                    source_layout,
                    coordinates,
                    state.clone(),
                    None,
                    topology.topology().world_size() > 1,
                ));
            Ok(PreparedPredictionExtension::DeepSeekV4 {
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(parameters),
                units,
                state,
                construction,
            })
        }
    }
    // Keep cold family constructors in separate frames; none needs another family's locals.
    #[inline(never)]
    pub(super) fn inkling(
        self,
        args: &crate::inkling::ModelArgs,
    ) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError> {
        let Self {
            extension: _,
            topology: _,
            tensor_rank,
            tasks,
            source_context,
            execution_context,
        } = self;
        let mut formats = args
            .text_config
            .quantized_weight_configs
            .clone()
            .unwrap_or_default();
        for task in tasks {
            match task.executable().weight_quantization() {
                Some(format) => {
                    formats.insert(task.name().to_owned(), format);
                }
                None => {
                    formats.remove(task.name());
                }
            }
        }
        let target_args =
            crate::inkling::with_checkpoint_formats(args, formats).map_err(invalid)?;
        let parameters =
            crate::inkling::LayeredModel::<B>::new(target_args.clone(), execution_context)
                .and_then(|model| model.parameter_description(execution_context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description,B::construction_metadata(execution_context))))
                .map_err(|error| invalid(error.to_string()))?;
        let layout =
            crate::partitioned_execution::derive_partitioned_local_layout(&parameters, tensor_rank)
                .map_err(invalid)?;
        let source_spec = crate::inkling::MtpModelSpec::new(args)
            .map_err(|error| invalid(error.to_string()))?
            .ok_or_else(|| invalid("Inkling prediction extension has no configured depth"))?;
        let local_spec = crate::inkling::MtpModelSpec::new(&target_args)
            .map_err(|error| invalid(error.to_string()))?
            .ok_or_else(|| invalid("Inkling prediction extension has no configured depth"))?;
        let source = source_spec
            .instantiate::<B>(source_context)
            .map_err(|error| invalid(error.to_string()))?;
        let local = local_spec
            .instantiate::<B>(execution_context)
            .map_err(|error| invalid(error.to_string()))?;
        let state = crate::inkling::mtp_state_layout(args)
            .map_err(|error| invalid(error.to_string()))?
            .ok_or_else(|| invalid("Inkling prediction extension has no state layout"))?;
        let shared = match (source.chain_norm, local.chain_norm) {
            (Some(source), Some(local)) => Some(PreparedPredictionUnit::new_shared(
                crate::inkling::MtpShared { chain_norm: source },
                crate::inkling::MtpShared { chain_norm: local },
                tasks,
            )?),
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "Inkling source and executable chain norms disagree",
                ));
            }
        };
        let units: Vec<_> = source
            .layers
            .into_iter()
            .zip(local.layers)
            .map(|(source, local)| PreparedPredictionUnit::new(source, local, tasks))
            .collect::<Result<_, _>>()?;
        let construction =
            std::sync::Arc::new(construction::PreparedPredictionConstruction::inkling(
                source_spec,
                local_spec,
                tasks.to_vec(),
                units.iter().map(|unit| unit.tasks.clone()).collect(),
                shared.as_ref().map(|unit| unit.tasks.clone()),
                state.clone(),
            ));
        Ok(PreparedPredictionExtension::Inkling {
            layout: std::sync::Arc::new(layout),
            parameters: std::sync::Arc::new(parameters),
            units,
            shared,
            state,
            construction,
        })
    }
    // Keep cold family constructors in separate frames; none needs another family's locals.
    #[inline(never)]
    pub(super) fn qwen(
        self,
        args: &crate::qwen::hybrid::ParsedHybridConfig,
    ) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError> {
        let Self {
            extension,
            topology,
            tensor_rank,
            tasks,
            source_context,
            execution_context,
        } = self;
        // Prediction units carry their own resident expert banks, populated
        // by the same exact parameter tasks as their attention and shared
        // projections. Execution uses ResidentExpertProvider, independently
        // of the target's expert residency policy.
        let description = if args.vision.is_some() {
            crate::qwen::hybrid::ConditionalLayeredModel::<B>::new(args.clone(), source_context)
                .and_then(|model| model.parameter_description(source_context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description,B::construction_metadata(source_context))))
        } else {
            crate::qwen::hybrid::LayeredModel::<B>::new(args.text.clone(), source_context)
                .and_then(|model| model.parameter_description(source_context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description,B::construction_metadata(source_context))))
        }
        .map_err(|error| invalid(error.to_string()))?;
        let layout = crate::partitioned_execution::derive_partitioned_local_layout(
            &description,
            tensor_rank,
        )
        .map_err(invalid)?;
        let geometry = crate::qwen::hybrid::local_geometry(&args.text, &layout)
            .map_err(|error| invalid(error.to_string()))?;
        let target = usize::try_from(args.text.num_hidden_layers)
            .map_err(|_| invalid("Qwen hybrid target count exceeds usize"))?;
        let descriptor =
            crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                extension.complete_architecture().clone(),
            )
            .architecture_descriptor();
        let mut units = Vec::with_capacity(extension.depth());
        let mut source_specs = Vec::with_capacity(extension.depth());
        let mut local_specs = Vec::with_capacity(extension.depth());
        let mut coordinate_rows = Vec::with_capacity(extension.depth());
        for depth in 0..extension.depth() {
            let source_spec = crate::qwen::hybrid::PredictionUnitSpec::new(&args.text, depth)
                .map_err(|error| invalid(error.to_string()))?;
            let source = source_spec
                .instantiate::<B>(source_context)
                .map_err(|error| invalid(error.to_string()))?;
            let local_config = geometry.prediction(depth).ok_or_else(|| {
                invalid(format!(
                    "Qwen hybrid prediction depth {depth} has no local geometry"
                ))
            })?;
            let local_spec = crate::qwen::hybrid::PredictionUnitSpec::new(local_config, depth)
                .map_err(|error| invalid(error.to_string()))?;
            let mut local = local_spec
                .instantiate::<B>(execution_context)
                .map_err(|error| invalid(error.to_string()))?;
            source_specs.push(source_spec);
            local_specs.push(local_spec);
            let mut coordinate_row = None;
            if let crate::qwen::hybrid::FeedForward::Routed(moe) = &mut local.block.feed_forward {
                let scope = descriptor.component_scopes.iter().find(|scope| matches!(
                    scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth: selected } if selected == depth
                )).ok_or_else(|| invalid("prepared Qwen prediction has no component scope"))?;
                let [component] = scope.routed_components.as_slice() else {
                    return Err(invalid(
                        "Qwen prediction must declare one resident routed bank",
                    ));
                };
                let coordinates = crate::component_partition::derive_coordinates_for_experts(
                    component,
                    &layout,
                    &(0..component.expert_count).collect::<Vec<_>>(),
                )
                .map_err(|error| invalid(error.to_string()))?;
                let coordinates = std::sync::Arc::new(coordinates.units().clone());
                coordinate_row = Some(coordinates.clone());
                moe.bind_shared_resident_unit_coordinates(
                    coordinates,
                    topology.topology().world_size() > 1,
                );
            }
            coordinate_rows.push(coordinate_row);
            units.push(PreparedPredictionUnit::new(source, local, tasks)?);
        }
        let state = geometry
            .state_layout()
            .slice(target..target + extension.depth())
            .map_err(|error| invalid(error.to_string()))?;
        let shared_spec = crate::qwen::hybrid::PredictionSharedSpec::new(
            args.text.hidden_size,
            args.text.rms_norm_eps,
        );
        let shared = PreparedPredictionUnit::new_shared(
            shared_spec
                .instantiate::<B>(source_context)
                .map_err(|error| invalid(error.to_string()))?,
            shared_spec
                .instantiate::<B>(execution_context)
                .map_err(|error| invalid(error.to_string()))?,
            tasks,
        )?;
        let construction = std::sync::Arc::new(construction::PreparedPredictionConstruction::qwen(
            source_specs,
            local_specs,
            shared_spec,
            tasks.to_vec(),
            units.iter().map(|unit| unit.tasks.clone()).collect(),
            shared.tasks.clone(),
            coordinate_rows,
            state.clone(),
            topology.topology().world_size() > 1,
        ));
        Ok(PreparedPredictionExtension::QwenHybrid {
            shared,
            layout: std::sync::Arc::new(layout),
            parameters: std::sync::Arc::new(description),
            units,
            state,
            construction,
        })
    }
    // Keep cold family constructors in separate frames; none needs another family's locals.
    #[inline(never)]
    pub(super) fn nemotron(
        self,
        args: &crate::nemotron_h::ModelArgs,
    ) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError> {
        let Self {
            extension,
            topology,
            tensor_rank,
            tasks,
            source_context,
            execution_context,
        } = self;
        let source_architecture =
            crate::nemotron_h::LayeredModel::<B>::new(args.clone(), source_context)
                .map_err(|error| invalid(error.to_string()))?;
        let source_description = source_architecture
            .parameter_description(source_context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description,B::construction_metadata(source_context)))
            .map_err(|error| invalid(error.to_string()))?;
        // Auxiliary tasks can lower prediction matrices independently of the
        // target. Preserve source formats elsewhere and construct executable
        // modules from the exact selected formats, including companions.
        let mut formats = source_description
            .groups()
            .iter()
            .flat_map(|group| group.group().members())
            .filter_map(|member| {
                args.weight_quantization_for(member.target())
                    .map(|format| (member.target().to_owned(), format))
            })
            .collect::<std::collections::HashMap<_, _>>();
        for task in tasks {
            if let Some(format) = task.executable().weight_quantization() {
                formats.insert(task.name().to_owned(), format);
            } else {
                formats.remove(task.name());
            }
        }
        let target_args =
            crate::nemotron_h::with_checkpoint_formats(args, formats).map_err(invalid)?;
        let description =
            crate::nemotron_h::LayeredModel::<B>::new(target_args.clone(), execution_context)
                .map_err(|error| invalid(error.to_string()))?
                .parameter_description(execution_context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description,B::construction_metadata(execution_context)))
                .map_err(|error| invalid(error.to_string()))?;
        let layout = crate::partitioned_execution::derive_partitioned_local_layout(
            &description,
            tensor_rank,
        )
        .map_err(invalid)?;
        let geometry = crate::nemotron_h::local_geometry(&target_args, &layout)
            .map_err(|error| invalid(error.to_string()))?;
        let source_layout = if tasks.iter().any(|task| {
            matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            )
        }) {
            Some(std::sync::Arc::new(
                crate::partitioned_execution::derive_partitioned_transform_source_layout(
                    &source_description,
                    &description,
                    tensor_rank,
                )
                .map_err(invalid)?,
            ))
        } else {
            None
        };
        let source_geometry = source_layout
            .as_ref()
            .map(|layout| crate::nemotron_h::local_geometry(args, layout))
            .transpose()
            .map_err(|error| invalid(error.to_string()))?;
        let policies = args
            .mtp_policies()
            .map_err(|error| invalid(error.to_string()))?;
        let pattern = policies
            .len()
            .checked_div(extension.depth())
            .filter(|pattern| *pattern > 0)
            .ok_or_else(|| invalid("Nemotron-H MTP pattern is empty"))?;
        let descriptor =
            crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                extension.complete_architecture().clone(),
            )
            .architecture_descriptor();
        let mut source_specs = Vec::with_capacity(policies.len());
        let mut local_specs = Vec::with_capacity(policies.len());
        let mut retained_rows = Vec::with_capacity(policies.len());
        let mut retained_coordinates = Vec::with_capacity(policies.len());
        let mut groups = Vec::with_capacity(extension.depth());
        for prediction in 0..extension.depth() {
            let mut units = Vec::with_capacity(pattern);
            for relative in 0..pattern {
                let physical = prediction * pattern + relative;
                let source_spec = match &source_geometry {
                    Some(geometry) => {
                        let geometry =
                            geometry.prediction_unit(physical).copied().ok_or_else(|| {
                                invalid(format!(
                                "Nemotron-H prediction source unit {physical} has no local geometry"
                            ))
                            })?;
                        crate::nemotron_h::PredictionUnitSpec::with_geometry(
                            args,
                            prediction,
                            relative,
                            policies[physical],
                            geometry,
                        )
                    }
                    None => crate::nemotron_h::PredictionUnitSpec::new(args, prediction, relative),
                }
                .map_err(|error| invalid(error.to_string()))?;
                let local_geometry =
                    geometry.prediction_unit(physical).copied().ok_or_else(|| {
                        invalid(format!(
                            "Nemotron-H prediction unit {physical} has no local geometry"
                        ))
                    })?;
                let local_spec = crate::nemotron_h::PredictionUnitSpec::with_geometry(
                    &target_args,
                    prediction,
                    relative,
                    policies[physical],
                    local_geometry,
                )
                .map_err(|error| invalid(error.to_string()))?;
                let source = source_spec
                    .instantiate::<B>(source_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let mut local = local_spec
                    .instantiate::<B>(execution_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let mut retained_coordinate = None;
                if let crate::nemotron_h::Operator::Sparse(moe) = &mut local.block.operator {
                    let scope = descriptor.component_scopes.iter().find(|scope| matches!(
                        scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth } if depth == prediction
                    )).ok_or_else(|| invalid("prepared Nemotron prediction has no component scope"))?;
                    let component = scope
                        .routed_components
                        .iter()
                        .find(|component| {
                            component.layer_index == args.num_hidden_layers as usize + physical
                        })
                        .ok_or_else(|| {
                            invalid("prepared Nemotron prediction has no declared routed bank")
                        })?;
                    let coordinates = crate::component_partition::derive_coordinates_for_experts(
                        component,
                        &layout,
                        &(0..component.expert_count).collect::<Vec<_>>(),
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    let coordinates = std::sync::Arc::new(coordinates.units().clone());
                    moe.bind_shared_resident_unit_coordinates(
                        coordinates.clone(),
                        topology.topology().world_size() > 1,
                    );
                    retained_coordinate = Some(coordinates);
                }
                let unit = PreparedPredictionUnit::new(source, local, tasks)?
                    .with_source_layout(source_layout.clone());
                retained_rows.push(unit.tasks.clone());
                retained_coordinates.push(retained_coordinate);
                source_specs.push(source_spec);
                local_specs.push(local_spec);
                units.push(unit);
            }
            groups.push(units);
        }
        let target = usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("Nemotron-H target depth exceeds usize"))?;
        let state = geometry
            .state_layout()
            .slice(target..target + policies.len())
            .map_err(|error| invalid(error.to_string()))?;
        let construction =
            std::sync::Arc::new(construction::PreparedPredictionConstruction::nemotron(
                source_specs,
                local_specs,
                tasks.to_vec(),
                retained_rows,
                source_layout,
                retained_coordinates,
                pattern,
                state.clone(),
                topology.topology().world_size() > 1,
            ));
        Ok(PreparedPredictionExtension::NemotronH {
            layout: std::sync::Arc::new(layout),
            parameters: std::sync::Arc::new(description),
            groups,
            state,
            construction,
        })
    }
}
