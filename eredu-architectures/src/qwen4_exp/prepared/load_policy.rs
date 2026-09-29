//! Architecture projection of explicit, backend-neutral bounded load policy.
use super::*;
use eredu_nn::TensorElementType;
use eredu_runtime::{AppendStreamBinding, BoundedExecutionPolicy, NormalizedLoadRequest};

/// A load policy cannot change retained equations or silently reduce admitted history.
#[derive(Debug, thiserror::Error)]
pub enum TargetLoadError {
    /// This family needs explicit invocation, row and stream bounds.
    #[error("qwen4_exp requires bounded execution policy")]
    MissingBounds,
    /// Managed or explicit execution limits cannot be represented.
    #[error(transparent)]
    Bounds(#[from] eredu_runtime::BoundedExecutionPolicyError),
    /// Portable cross-field validation failed before source or native work.
    #[error(transparent)]
    Request(#[from] eredu_runtime::NormalizedLoadRequestError),
    /// A partition request needs the separate architecture partition construction path.
    #[error("qwen4_exp replicated target load cannot consume a partitioned request")]
    PartitionPreparationRequired,
    /// Prediction intent needs joint target/prediction preparation, not a target-only route.
    #[error("qwen4_exp prediction intent requires joint prepared execution")]
    PredictionPreparationRequired,
    /// Explicit media intent must be consumed by joint target/vision preparation.
    #[error("qwen4_exp media intent requires joint conditional preparation")]
    MediaPreparationRequired,
    /// Low-level joint preparation requires explicit media representation intent.
    #[error("qwen4_exp conditional preparation requires explicit media execution policy")]
    MissingMediaPolicy,
    /// Prepared construction geometry differs from the requested execution bounds.
    #[error("qwen4_exp retained target limits differ from requested execution bounds")]
    RetainedLimits,
    /// State arithmetic must be an admitted floating representation.
    #[error("qwen4_exp state requires F16, BF16 or F32")]
    StateRepresentation,
    /// Finite policy cannot contain the declared family requirement.
    #[error("qwen4_exp {resource} needs {required}, limit is {limit}")]
    Budget {
        /// Resource measured in records or tokens.
        resource: &'static str,
        /// Required extent.
        required: u64,
        /// Explicit admitted bound.
        limit: u64,
    },
    /// Paged state and append streams must select the same physical page geometry.
    #[error("qwen4_exp append page size differs from selected state pages")]
    PageGeometry,
    /// Shared ordinary and bank selection policy cannot be represented.
    #[error(transparent)]
    Selection(#[from] crate::routed_text::RoutedTextSelectionError),
    /// Exact source, provider or state admission failed.
    #[error(transparent)]
    Preparation(#[from] PreparationError),
}

fn policy(request: &NormalizedLoadRequest) -> Result<BoundedExecutionPolicy, TargetLoadError> {
    request.validate_model_preparation()?;
    request
        .bounded_execution()
        .ok_or(TargetLoadError::MissingBounds)
}

impl TargetLimits {
    /// Derives architecture limits before preparing either source format. Floating
    /// representation comes from admitted parameter metadata, not the backend family name.
    pub fn from_load_request(
        request: &NormalizedLoadRequest,
        element: TensorElementType,
    ) -> Result<Self, TargetLoadError> {
        let policy = policy(request)?;
        if !matches!(
            element,
            TensorElementType::F16 | TensorElementType::Bf16 | TensorElementType::F32
        ) {
            return Err(TargetLoadError::StateRepresentation);
        }
        let invocation = policy.invocation();
        let selection = policy.selection();
        Ok(Self {
            qsa: super::super::qsa::QsaExecutionLimits {
                batch: invocation.batch(),
                tokens: invocation.chunk_tokens(),
                workspace_bytes: selection.invocation_workspace_bytes(),
            },
            history_tokens: invocation.history_tokens(),
            tile_blocks: selection.tile_entries(),
            selection_workspace: selection.selection_workspace_bytes(),
            invocation_tokens: invocation.invocation_tokens(),
            lookup_rows: policy.rows().limits().requests,
            element,
        })
    }

    /// Checks a cumulative causal prefix before embedding, state writes or transport resume.
    pub(crate) fn validate_history(
        &self,
        offset: i32,
        tokens: i32,
    ) -> Result<(), super::super::input::RequestError> {
        use super::super::input::RequestError;
        if offset < 0 || tokens <= 0 || self.history_tokens <= 0 {
            return Err(RequestError::Boundary);
        }
        let required = offset as u64 + tokens as u64;
        if required > self.history_tokens as u64 {
            return Err(RequestError::History {
                required,
                limit: self.history_tokens as u64,
            });
        }
        Ok(())
    }
}

/// Shared family projection of a normalized request, independent of source binding.
pub(super) struct TargetLoadProjection {
    pub limits: TargetLimits,
    pub selection: crate::routed_text::RoutedTextSelectionRequest,
    policy: BoundedExecutionPolicy,
}

impl TargetLoadProjection {
    pub fn new(
        request: &NormalizedLoadRequest,
        config: &Config,
        element: TensorElementType,
    ) -> Result<Self, TargetLoadError> {
        Self::project(request, config, element, false)
    }

    pub(super) fn partitioned(
        request: &NormalizedLoadRequest,
        config: &Config,
        element: TensorElementType,
    ) -> Result<Self, TargetLoadError> {
        Self::project(request, config, element, true)
    }

    fn project(
        request: &NormalizedLoadRequest,
        config: &Config,
        element: TensorElementType,
        partitioned: bool,
    ) -> Result<Self, TargetLoadError> {
        let policy = policy(request)?;
        if matches!(
            request.media_execution(),
            eredu_runtime::MediaLoadRequest::Required(_)
        ) {
            return Err(TargetLoadError::MediaPreparationRequired);
        }
        if request.has_parallel_execution() != partitioned {
            return Err(TargetLoadError::PartitionPreparationRequired);
        }
        if matches!(
            request.drafting(),
            eredu_runtime::DraftingLoadRequest::Embedded { .. }
        ) {
            return Err(TargetLoadError::PredictionPreparationRequired);
        }
        let limits = TargetLimits::from_load_request(request, element)?;
        limits
            .validate_for_config(config)
            .map_err(PreparationError::from)?;
        let append = policy.append();
        if let eredu_runtime::CacheResidencyPolicy::Paged(pages) = request.state_residency() {
            if append.limits().page_entries != pages.block_size_tokens() as usize {
                return Err(TargetLoadError::PageGeometry);
            }
        }
        // Only complete visible micro-blocks occupy append streams; the partial
        // tail belongs to fixed state, independent of prefill chunk boundaries.
        let required = limits.history_tokens as u64 / config.attention.ratio as u64;
        if required > append.limits().entries as u64 {
            return Err(TargetLoadError::Budget {
                resource: "summary history records",
                required,
                limit: append.limits().entries as u64,
            });
        }
        let selection = crate::routed_text::RoutedTextSelectionRequest::new(
            request
                .validate_model_preparation()?
                .text_selection_request(request.required_session_capabilities(), partitioned),
            request.weight_residency(),
        )?;
        Ok(Self {
            limits,
            selection,
            policy,
        })
    }

    pub fn streams(&self, spec: &TargetSpec) -> Vec<AppendStreamBinding> {
        let append = self.policy.append();
        spec.units
            .iter()
            .enumerate()
            .flat_map(|(layer, unit)| match unit {
                UnitSpec::Decoder {
                    mixer: super::super::target::MixerSpec::Indexed(attention),
                    ..
                } => attention
                    .state
                    .streams()
                    .into_iter()
                    .map(|spec| AppendStreamBinding {
                        layer,
                        lanes: self.limits.qsa.batch as u32,
                        spec,
                        limits: append.limits(),
                        payload_bytes: append.payload_bytes(),
                        scratch_bytes: append.scratch_bytes(),
                        catalog_bytes: append.catalog_bytes(),
                    })
                    .collect::<Vec<_>>(),
                _ => vec![],
            })
            .collect()
    }

    pub fn row_limits(&self) -> RowLookupLimits {
        self.policy.rows().limits()
    }

    pub fn select_rows(
        &self,
        descriptors: eredu_runtime::RowLookupDescriptors,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
    ) -> Result<eredu_runtime::SelectedRowLookupPlans, TargetLoadError> {
        eredu_runtime::SelectedRowLookupPlans::select(
            descriptors,
            self.policy.rows().bank(),
            self.policy.rows().retained_scalar_bytes(),
            support,
        )
        .map_err(PreparationError::from)
        .map_err(TargetLoadError::from)
    }
}

impl PreparedTarget {
    /// Projects generic load bounds onto this exact retained replicated target.
    /// Headers and bound targets share the same policy projection; this method
    /// neither reopens sources nor chooses native mechanisms.
    pub fn execution_plan_for_load(
        &self,
        request: &NormalizedLoadRequest,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
    ) -> Result<TargetExecutionPlan, TargetLoadError> {
        let projection =
            TargetLoadProjection::new(request, &self.spec.config, self.spec.limits.element)?;
        if projection.limits != self.spec.limits {
            return Err(TargetLoadError::RetainedLimits);
        }
        let rows = self.row_lookups(projection.row_limits(), ResidencyPolicy::Cacheable)?;
        let admission = projection.select_rows(rows.descriptors().clone(), support)?;
        let mut plan = self.execution_plan(projection.streams(&self.spec), admission)?;
        plan.load_selection = Some(projection.selection);
        Ok(plan)
    }
}
