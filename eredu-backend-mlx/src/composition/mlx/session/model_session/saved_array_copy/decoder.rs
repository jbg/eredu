//! One source-bound decoder/sampler copy using the existing submission engine.

use super::*;
pub(in crate::composition::mlx::session) use owner::CopiedTextComponentsOwner;
use owner::FrozenDecoderOwner;

pub(in crate::composition::mlx::session) mod cold_source;
mod finite_preparation;
mod original;
mod owner;
mod preparation;
pub(in crate::composition::mlx::session) mod resume_driver;
mod resume_estimate;
mod resume_prompt;
pub(in crate::composition::mlx::session) use resume_estimate::LogicalResumeSource;
mod resume_quote;
mod storage;
use crate::backend::runtime::cache::state::{
    InitializedResidentDecoderCopy, PreparedResidentDecoderCopy, SavedResidentDecoderCopy,
};
use eredu_runtime::replicated_session::ReplicatedTextControlOrigin;
use eredu_runtime::working_memory::WorkingMemoryStorage;
use eredu_runtime::{SharedHostMetadata, SharedPreparedInputCacheIdentity};
pub(in crate::composition::mlx::session) use resume_quote::{PreparedSavedTextResumeQuote, ResumeCapture};

/// Temporary access custody. The saved result never retains a live executable.
#[derive(Clone)]
enum DecoderCopyOwner {
    Live(SessionPayloadOwner),
    Saved(FrozenDecoderOwner),
}

impl DecoderCopyOwner {
    fn prepare(&self) -> Result<PreparedResidentDecoderCopy<'_>, Error> {
        match self {
            Self::Live(payload) => payload.model.erased().prepare_resident_decoder_copy(),
            Self::Saved(saved) => saved.native.prepare_copy(),
        }
    }

    fn prepare_fixed(
        &self,
    ) -> Result<
        PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    > {
        match self {
            Self::Live(payload) => payload.model.erased().prepare_resident_decoder_copy_fixed(),
            Self::Saved(saved) => saved.native.prepare_copy_fixed(),
        }
    }

    fn control_origin(&self) -> Result<ReplicatedTextControlOrigin, Error> {
        match self {
            Self::Live(payload) => payload.model.erased().resident_control_origin(),
            Self::Saved(saved) => Ok(saved.origin.clone()),
        }
    }

    fn input_identity(&self) -> Result<Option<SharedPreparedInputCacheIdentity>, Error> {
        match self {
            Self::Live(payload) => payload.model.erased().resident_copy_input_identity(),
            Self::Saved(saved) => Ok(saved.input.clone()),
        }
    }

    fn validate(&self, runtime: &ModelRuntime<MlxBackend<'_>>) -> Result<(), Error> {
        if let Self::Live(payload) = self {
            if !payload.same_owner(&runtime.session().payload) {
                return Err(mismatch());
            }
        }
        Ok(())
    }
}

struct FrozenDecoder {
    native: SavedResidentDecoderCopy,
    origin: ReplicatedTextControlOrigin,
    input: Option<SharedPreparedInputCacheIdentity>,
    // The queued retirement Box is destroyed before this last preparation token.
    _host_preparation: Option<eredu_core::HostPreparationAuthority>,
}

/// Recovery owns both actual source values and every partial native descriptor.
/// The account separately pins their registered allocation origins.
pub(in crate::composition::mlx::session::model_session) struct CopyRecovery {
    roots: Rc<RefCell<Vec<Array>>>,
    source: RefCell<Option<CopyRecoverySource>>,
}

// The prompt worker retains the whole authenticated pair, including its host
// sampler, while the existing immutable-copy path keeps its original custody.
enum CopyRecoverySource {
    Decoder(DecoderCopyOwner),
    Resume(CopiedTextComponentsOwner),
}

impl CopyRecovery {
    fn retire(&self) {
        let roots = std::mem::take(&mut *self.roots.borrow_mut());
        let source = self.source.borrow_mut().take();
        drop(roots);
        match source {
            Some(CopyRecoverySource::Decoder(source)) => drop(source),
            Some(CopyRecoverySource::Resume(source)) => drop(source),
            None => {}
        }
    }
}

pub(in crate::composition::mlx::session) struct PreparedTextComponentsCopy<'a> {
    sampling: TextArrayBinding<'a>,
    decoder: DecoderCopyOwner,
    source_native: Option<crate::backend::nn::workspace::ProjectedNativeStorage>,
    proof: RegisteredWorkspaceCopy<StorageIdentity>,
    complete_source: WorkingMemoryStorage<StorageIdentity>,
    required_bytes: u64,
    original: Option<original::CopyRequirements>,
    physical_extra: u64,
    collector: RootCollectorPlan,
    destination: TextCopyDestination,
    // The preparation owner outlives every source/proof/table metadata field.
    host_preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl<'a> PreparedTextComponentsCopy<'a> {
    pub(in crate::composition::mlx::session) fn prepare(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &'a super::super::super::generation::MlxTextSamplingState,
        pending: Option<&'a MlxTextToken>,
    ) -> Result<Self, Error> {
        Self::prepare_with_host(runtime, sampling, pending, None)
    }

    pub(in crate::composition::mlx::session) fn prepare_with_host(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &'a super::super::super::generation::MlxTextSamplingState,
        pending: Option<&'a MlxTextToken>,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<Self, Error> {
        Self::prepare_with_input(
            runtime,
            sampling,
            pending.map(eredu_core::PendingTextInput::Decode),
            host,
        )
    }

    pub(in crate::composition::mlx::session) fn prepare_with_input(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &'a super::super::super::generation::MlxTextSamplingState,
        pending: Option<eredu_core::PendingTextInput<&'a MlxModelInput, &'a MlxTextToken>>,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<Self, Error> {
        Self::prepare_with_input_and_capture(runtime, sampling, pending, host, None)
    }

    pub(in crate::composition::mlx::session) fn prepare_with_input_and_capture(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &'a super::super::super::generation::MlxTextSamplingState,
        pending: Option<eredu_core::PendingTextInput<&'a MlxModelInput, &'a MlxTextToken>>,
        host: Option<&eredu_core::HostPreparationAuthority>,
        capture: Option<&super::capture::SavedCaptureCheckpoint>,
    ) -> Result<Self, Error> {
        let capture_source = capture.map(|capture| capture.checkpoint().source());
        let result = (|| match pending {
            Some(eredu_core::PendingTextInput::Prefill(prompt)) => {
                let authority = host.ok_or_else(unknown)?;
                cold_source::inspect_live(runtime, sampling, None, capture_source)
                    .map_err(|cause| cold_source::retain_failure(cause, authority))?;
                let prompt = super::pending_input::PromptCopySource::prepare(sampling, prompt)
                    .map_err(|cause| cold_source::retain_failure(cause.into(), authority))?;
                let source =
                    TextArraySource::inspect_with_capture(runtime, sampling, None, host, capture)?;
                Self::prepare_source(
                    runtime,
                    TextArrayBinding::LivePrefill {
                        sampling,
                        prompt,
                        source,
                    },
                    DecoderCopyOwner::Live(runtime.session().payload.clone()),
                    host,
                )
            }
            pending => {
                let pending = match pending {
                    Some(eredu_core::PendingTextInput::Decode(token)) => Some(token),
                    None => None,
                    Some(eredu_core::PendingTextInput::Prefill(_)) => unreachable!(),
                };
                if let Some(authority) = host {
                    cold_source::inspect_live(runtime, sampling, pending, capture_source)
                        .map_err(|cause| cold_source::retain_failure(cause, authority))?;
                }
                Self::prepare_live_inner(runtime, sampling, pending, host, capture)
            }
        })();
        match host {
            Some(authority) => {
                result.map_err(|cause| cold_source::retain_operation_failure(cause, authority))
            }
            None => result,
        }
    }

    fn prepare_live_inner(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &'a super::super::super::generation::MlxTextSamplingState,
        pending: Option<&'a MlxTextToken>,
        host: Option<&eredu_core::HostPreparationAuthority>,
        capture: Option<&super::capture::SavedCaptureCheckpoint>,
    ) -> Result<Self, Error> {
        runtime.session().validate_backend(runtime.backend())?;
        runtime
            .session()
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let source =
            TextArraySource::inspect_with_capture(runtime, sampling, pending, host, capture)?;
        Self::prepare_source(
            runtime,
            TextArrayBinding::Live {
                sampling,
                pending,
                source,
            },
            DecoderCopyOwner::Live(runtime.session().payload.clone()),
            host,
        )
    }

    pub(in crate::composition::mlx::session) fn prepare_saved(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        saved: &'a CopiedTextComponents,
    ) -> Result<Self, Error> {
        Self::prepare_source(
            runtime,
            TextArrayBinding::Saved(&saved.sampling),
            DecoderCopyOwner::Saved(saved.decoder.clone()),
            None,
        )
    }

    fn prepare_source(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: TextArrayBinding<'a>,
        decoder: DecoderCopyOwner,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<Self, Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        session
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        sampling.validate(runtime)?;
        decoder.validate(runtime)?;
        let lease = session
            .authority
            .borrow_mut()
            .begin_submission()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let plan = match host {
            Some(authority) => decoder.prepare_fixed().map_err(|cause| {
                cold_source::retain_failure(
                    cold_source::SourcePreparationCause::Decoder(cause),
                    authority,
                )
            })?,
            None => decoder.prepare()?,
        };
        let pool = runtime.backend().memory_pool();
        let collector_for = |operand_count: usize, host_rows: usize| {
            let mut source_count =
                Some(sampling.key().iter().count() + sampling.pending().iter().count());
            if plan.is_paged() {
                crate::backend::runtime::cache::state::SnapshotArraySources::visit_operands(
                    &plan,
                    &mut |source| {
                        if matches!(
                            source,
                            crate::backend::runtime::cache::state::SnapshotOperand::Array(_)
                        ) {
                            source_count = source_count.and_then(|n| n.checked_add(1));
                        }
                        Ok(())
                    },
                )
                .map_err(
                    crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
                )?;
            } else {
                plan.visit_retained_arrays(&mut |_| {
                    source_count = source_count.and_then(|n| n.checked_add(1))
                })
                .map_err(
                    crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
                )?;
            }
            let source_count = source_count.ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
            let roots = operand_count
                .checked_mul(2)
                .and_then(|n| n.checked_add(source_count))
                .and_then(|n| {
                    host_rows
                        .checked_mul(2)
                        .and_then(|extra| n.checked_add(extra))
                })
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
            let collector = RootCollectorPlan::from_root_count(roots)?;
            Ok::<_, Error>(collector)
        };
        let (program, registered, complete_source, collector, original, source_native) =
            if let Some(host) = host {
                let mechanisms = session
                    .payload
                    .model
                    .workspace_mechanisms()
                    .ok_or_else(unknown)?;
                let finite = finite_preparation::SnapshotFinitePreparation::inspect(
                    &decoder,
                    &plan,
                    sampling.key(),
                    sampling.pending(),
                    pool,
                    mechanisms,
                    runtime.backend(),
                )
                .map_err(|cause| finite_preparation::retain_failure(cause, host))?;
                let prepared = finite.construct(host)?;
                let collector = collector_for(
                    prepared.operand_count,
                    prepared.copy_requirements.host_rows(),
                )?;
                (
                    prepared.program,
                    prepared.registered,
                    prepared.complete_source,
                    collector,
                    Some(prepared.copy_requirements),
                    Some(prepared.source_native),
                )
            } else {
                let source = decoder.complete_storage()?;
                let complete_source = source.pin_registered(pool)?;
                let mechanisms = session
                    .payload
                    .model
                    .resident_workspace_mechanisms()
                    .ok_or_else(unknown)?;
                let context = WorkspaceContext::new(mechanisms);
                let mut operand_count = Some(
                    usize::from(sampling.key().is_some())
                        + usize::from(sampling.pending().is_some()),
                );
                plan.visit_operands(&mut |_| {
                    operand_count = operand_count.and_then(|count| count.checked_add(1))
                })
                .map_err(
                    crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
                )?;
                let operand_count =
                    operand_count.ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
                let mut native = None;
                let mut projection = crate::backend::nn::workspace::OwnedArrayProjection::prepare(
                    &mut native,
                    &context,
                    operand_count,
                )
                .map_err(|cause| Error::Other(Box::new(cause)))?;
                let mut inputs = Vec::new();
                inputs
                    .try_reserve_exact(operand_count)
                    .map_err(|cause| Error::Other(Box::new(cause)))?;
                let mut failure = None;
                let mut project = |array: &Array| {
                    if failure.is_none() {
                        match projection.project_prepared(array) {
                            Ok(input) => inputs.push(input),
                            Err(cause) => failure = Some(cause),
                        }
                    }
                };
                plan.visit_operands(&mut project).map_err(
                    crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
                )?;
                for array in sampling.key().into_iter().chain(sampling.pending()) {
                    project(array);
                }
                if let Some(cause) = failure {
                    return Err(Error::Other(Box::new(cause)));
                }
                drop(projection);
                let native = native.expect("installed owned projection");
                let collector = collector_for(inputs.len(), 0)?;
                if !native.is_complete() {
                    return Err(unknown());
                }
                let registered = RegisteredWorkspaceStorage::bind(
                    pool,
                    &context,
                    native
                        .iter()
                        .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone())),
                )
                .map_err(memory)?;
                let program = WorkspaceIsolatedCopyPlan::prepare(
                    &context,
                    registered.borrowed_storage(),
                    &inputs,
                )
                .map_err(|error| Error::Other(Box::new(error)))?;
                (program, registered, complete_source, collector, None, None)
            };
        let numerical_bytes = program.incremental_bytes().ok_or_else(unknown)?;
        let host_bytes = sampling
            .sampler()
            .prepare_copy()
            .map_err(|error| Error::Other(Box::new(error)))?
            .retained_bytes();
        let decoder_bytes = match host {
            Some(authority) => plan
                .host_copy_initialization_peak_bytes()
                .map_err(|cause| {
                    cold_source::retain_failure(
                        cold_source::SourcePreparationCause::Decoder(cause),
                        authority,
                    )
                })?,
            None => plan.host_copy(pool)?.initialization_peak_bytes(),
        };
        let physical_extra = original
            .map(|requirements| requirements.physical_extra(numerical_bytes))
            .transpose()
            .map_err(memory)?
            .unwrap_or(0);
        let native_controls = collector.copy_control_bytes()?;
        let required_bytes = host_bytes
            .checked_add(decoder_bytes)
            .and_then(|n| n.checked_add(numerical_bytes))
            .and_then(|n| n.checked_add(physical_extra))
            .and_then(|n| n.checked_add(native_controls))
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let proof = RegisteredWorkspaceCopy::bind(program, registered)
            .map_err(|error| Error::Other(Box::new(error)))?;
        Ok(Self {
            sampling,
            decoder,
            source_native,
            proof,
            complete_source,
            required_bytes,
            original,
            physical_extra,
            collector,
            destination: TextCopyDestination {
                session: Rc::clone(&session.poison),
                lease,
            },
            host_preparation: host.cloned(),
        })
    }

    pub(in crate::composition::mlx::session) fn required_bytes(&self) -> u64 {
        self.required_bytes
    }

    pub(in crate::composition::mlx::session) fn copy(
        self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        limits: WorkspaceCopyLimits,
    ) -> Result<CopiedTextComponents, Error> {
        let host = self.host_preparation.clone();
        let result = self.copy_inner(runtime, limits);
        match host {
            Some(authority) => {
                result.map_err(|cause| cold_source::retain_operation_failure(cause, &authority))
            }
            None => result,
        }
    }

    fn copy_inner(
        self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        limits: WorkspaceCopyLimits,
    ) -> Result<CopiedTextComponents, Error> {
        self.destination.validate(runtime)?;
        self.sampling.validate(runtime)?;
        self.decoder.validate(runtime)?;
        let plan = match &self.host_preparation {
            Some(authority) => self.decoder.prepare_fixed().map_err(|cause| {
                cold_source::retain_failure(
                    cold_source::SourcePreparationCause::Decoder(cause),
                    authority,
                )
            })?,
            None => self.decoder.prepare()?,
        };
        let input = self.decoder.input_identity()?;
        let origin = self.decoder.control_origin()?;
        let (temperature, next_prediction, parameter_epoch) = self.sampling.metadata();
        let pool = runtime.backend().memory_pool();
        // Recount the same pinned source before admitting any destination. The
        // borrowed environment is consumed into owned native controls before
        // taking the mutable session loan.
        let native_error = |cause| {
            cold_source::retain_failure(
                cold_source::SourcePreparationCause::NativeCopy(cause),
                self.host_preparation
                    .as_ref()
                    .expect("original requirements retain H"),
            )
        };
        let environment = if self
            .original
            .is_some_and(original::CopyRequirements::has_native)
        {
            Some(
                runtime
                    .backend()
                    .original_copy_environment()
                    .map_err(original::CopyPreparationCause::from)
                    .map_err(native_error)?,
            )
        } else {
            None
        };
        let native_plan = environment
            .as_ref()
            .map(|environment| {
                original::native_plan(
                    &plan,
                    self.sampling.key(),
                    self.sampling.pending(),
                    environment,
                )
            })
            .transpose()
            .map_err(original::CopyPreparationCause::from)
            .map_err(native_error)?
            .flatten();
        let publication = self
            .original
            .map(|requirements| {
                requirements.validate_native(native_plan.as_ref())?;
                requirements.publication_plan()
            })
            .transpose()
            .map_err(memory)?;
        let initialized = if native_plan.is_some() {
            Some(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .prefill_roots_runtime()?,
            )
        } else {
            None
        };
        let sampling = RegisteredSamplingCopy::prepare(self.sampling.borrow_funded()?, self.proof)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let host_copy = match &self.host_preparation {
            Some(authority) => plan.host_copy_prepared(pool, authority)?,
            None => plan.host_copy(pool)?,
        };
        let mut limits = limits;
        // Native physical_bytes is the whole copy envelope. Only its positive
        // delta above the shared numerical proof enters the same B account.
        limits.safety_reserve_bytes = limits
            .safety_reserve_bytes
            .checked_add(self.physical_extra)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let (sampler, mut slots, account) = host_copy.admit(
            pool,
            sampling,
            self.complete_source,
            self.collector.limits(limits)?,
        )?;
        let (custody, scope) = account.into_parts();
        let roots = match self.collector.construct() {
            Ok(roots) => roots,
            Err(cause) => {
                drop((slots, sampler));
                return Err(collector::allocation_failure(cause, custody, scope));
            }
        };
        let mut prepared = match native_plan
            .map(|plan| {
                plan.prepare(
                    &custody,
                    initialized
                        .as_ref()
                        .expect("native plan has initialization witness"),
                )
            })
            .transpose()
        {
            Ok(prepared) => prepared,
            Err(cause) => {
                // Native owner construction has not opened a Scope. Its error
                // retains the copy account through every partial prefix.
                drop((slots, sampler));
                let _ = scope.certify();
                return Err(cause.into());
            }
        };
        if let (Some(copy), Some(environment)) = (prepared.as_mut(), environment.as_ref()) {
            if let Err(cause) = slots.prepare_host_destinations(copy, environment) {
                // No native submission has begun. Retire all constructed
                // destinations/rows before certifying this untouched scope.
                drop((slots, sampler, prepared));
                let _ = scope.certify();
                return Err(cause);
            }
        }
        drop(environment);
        let funding = match publication {
            Some(publication) => match publication.construct(
                scope,
                &custody,
                self.host_preparation
                    .as_ref()
                    .expect("original publication retains H"),
                prepared.as_ref().map(|prepared| prepared.budget()),
            ) {
                Ok(funding) => funding,
                Err(cause) => {
                    // The constructor retires/certifies its untouched scope;
                    // final empty tables and sampler retire before its error.
                    drop((slots, sampler));
                    return Err(cause);
                }
            },
            None => text_funding::FundedWork::new(scope),
        };
        let original = self.original.is_some();
        let host_only = original && prepared.is_none();
        let (backend, session) = runtime.parts_mut();
        // Ordinary stream cloning preserves its existing ownership. The
        // original worker uses the exact admitted environment stream loan.
        let ordinary_stream = (!original).then(|| backend.stream().clone());
        let stream = ordinary_stream.as_ref().unwrap_or_else(|| backend.stream());
        if !original {
            original::retain_sources(&plan, &self.sampling, &roots, false, None)?;
        }
        let mut execution = None;
        let mut submission = None;
        let (native, key, pending) = if host_only {
            // Empty native programs still copy host state and use the same
            // finite publication path, without creating a native Scope.
            copy_native(
                plan,
                slots,
                &self.sampling,
                stream,
                &roots,
                &funding,
                self.host_preparation.as_ref(),
            )?
        } else {
            let owner = SubmissionResources::with_purpose(
                self.destination.lease,
                Rc::clone(&session.poison),
                SubmissionPurpose::SavedComponentsCopy(CopyRecovery {
                    roots: Rc::clone(&roots),
                    source: RefCell::new(Some(CopyRecoverySource::Decoder(self.decoder.clone()))),
                }),
            );
            owner.payload.replace(Some(session.payload.clone()));
            owner.funding.replace(Some(funding.clone()));
            let recovery = match prepared {
                Some(prepared) => {
                    let (recovery, native) = prepared.begin(owner.ticket())?;
                    execution = Some(native);
                    recovery
                }
                None => owner.recovery()?,
            };
            let operation = SessionOperation {
                session,
                owner: owner.clone(),
                recovery: Some(recovery),
                handed_off: false,
                token_validations: Default::default(),
            };
            // The submission and original bank now retain every failure path,
            // including a refusal while cloning the first source descriptor.
            let copied = original::during_construction(&mut execution, || {
                if original {
                    original::retain_sources(
                        &plan,
                        &self.sampling,
                        &roots,
                        true,
                        self.source_native.as_ref(),
                    )?;
                }
                copy_native(
                    plan,
                    slots,
                    &self.sampling,
                    stream,
                    &roots,
                    &funding,
                    self.host_preparation.as_ref(),
                )
            });
            let copied = finish_saved_copy(operation, copied)?;
            submission = Some(owner);
            copied
        };
        // Keep execution (and its original buffer witness) alive until the
        // complete result has been registered and the neutral scope certified.
        let mut host = funding.prepare_inventory()?;
        if let Some(authority) = &self.host_preparation {
            let destination = original::destination_plan(&native, authority)?;
            let mut failure = None;
            destination.visit_registered_child_metadata_borrowed(&mut |metadata| {
                if failure.is_none() {
                    failure = host.include_slot_metadata(metadata.clone()).err();
                }
            });
            if let Some(cause) = failure {
                return Err(cause.into());
            }
        } else {
            native.visit_registered_child_metadata(&mut |metadata| {
                host.include_slot_metadata(metadata.clone())
                    .map_err(Error::from)
            })?;
        }
        funding.publish(host)?;
        if let Some(owner) = &submission {
            if let SubmissionPurpose::SavedComponentsCopy(recovery) = &owner.purpose {
                recovery.retire();
            }
        }
        funding.certify()?;
        drop(execution);
        Ok(CopiedTextComponents {
            decoder: FrozenDecoderOwner::new(FrozenDecoder {
                native,
                origin,
                input,
                _host_preparation: self.host_preparation.clone(),
            }),
            sampling: CopiedTextSampling {
                sampler,
                arrays: CopiedTextArrays {
                    key,
                    pending,
                    pending_metadata: self.sampling.pending_metadata(),
                    source: self.sampling.provenance(),
                    custody,
                },
                temperature,
                next_prediction,
                parameter_epoch,
            },
            host_preparation: self.host_preparation.clone(),
        })
    }
}

fn copy_native<'a>(
    plan: PreparedResidentDecoderCopy<'a>,
    slots: InitializedResidentDecoderCopy<'a>,
    sampling: &TextArrayBinding<'_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    funding: &text_funding::FundedWork,
    host: Option<&eredu_core::HostPreparationAuthority>,
) -> Result<(SavedResidentDecoderCopy, Option<Array>, Option<Array>), Error> {
    let paged = plan.is_paged();
    let validate = |array: &Array| -> Result<(), Error> {
        if host.is_some() {
            original::validate_completed_destination(array)
        } else {
            array.evaluated()?;
            Ok(())
        }
    };
    let mut observed = |array: &Array| {
        funding.retain(array);
        if let Some(cause) = funding.take_collection_failure() {
            return Err(cause);
        }
        validate(array)
    };
    let mut observe_host = |copy: &crate::backend::array_copy::PreparedSavedHostCopy| {
        funding.bind_snapshot_host_copy(copy)?;
        let output = copy
            .completed_output()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        funding.retain(output);
        if let Some(cause) = funding.take_collection_failure() {
            return Err(cause);
        }
        Ok(())
    };
    let native =
        plan.copy_retained_with_host(slots, stream, roots, &mut observed, &mut observe_host)?;
    let mut failure = None;
    let mut retain = |array: &Array| {
        if failure.is_none() {
            funding.retain(array);
            failure = validate(array).err();
        }
    };
    if !paged {
        let destination = match host {
            Some(authority) => original::destination_plan(&native, authority)?,
            None => native.prepare_copy()?,
        };
        if host.is_some() {
            // Compressed storage aliases share these logical destination backings;
            // the finite publisher inspects each actual copied operand once.
            destination.visit_operands(&mut retain).map_err(
                crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
            )?;
        } else {
            destination.visit_retained_arrays(&mut retain).map_err(
                crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
            )?;
        }
        drop(destination);
    }
    if let Some(error) = failure {
        return Err(error.into());
    }
    let one = |array: &Array| -> Result<Array, Error> {
        let copy = IsolatedArrayCopy::new(array).copy_retained(stream, roots)?;
        funding.retain(&copy);
        validate(&copy)?;
        Ok(copy)
    };
    let key = sampling.key().map(one).transpose()?;
    #[cfg(all(
        test,
        target_vendor = "apple",
        feature = "metal",
        not(feature = "cuda")
    ))]
    tests::after_key_copy()?;
    let pending = sampling.pending().map(one).transpose()?;
    Ok((native, key, pending))
}

/// Both components arise from one source and account. No field exports an
/// installable state, live request, lease, receipt or populated run retention.
pub(in crate::composition::mlx::session) struct CopiedTextComponents {
    decoder: FrozenDecoderOwner,
    sampling: CopiedTextSampling,
    // Returned aggregate wrappers and aliases retain the accepted H after payloads.
    host_preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl CopiedTextComponents {
    pub(in crate::composition::mlx::session) fn capture_checkpoint(
        &self,
    ) -> Option<&eredu_runtime::capture::FundedCaptureCheckpoint> {
        self.sampling
            .arrays
            .source
            .capture
            .as_ref()
            .map(|source| source.checkpoint())
    }

    pub(in crate::composition::mlx::session) fn capture_selection(
        &self,
    ) -> Option<&eredu_runtime::layered::PreparedCaptureSelection> {
        self.sampling
            .arrays
            .source
            .capture
            .as_ref()
            .map(|source| source.selection())
    }

    pub(in crate::composition::mlx::session) fn capture_witness(
        &self,
    ) -> Option<&eredu_runtime::working_memory::RegisteredInferenceSourceWitness> {
        self.sampling
            .arrays
            .source
            .capture
            .as_ref()
            .map(|source| source.witness())
    }

    /// Resume needs the exact executable origin; immutable duplication does not.
    /// Keep this separate from data-copy validation so a saved source remains
    /// independently copyable after its original session has retired.
    pub(in crate::composition::mlx::session) fn validate_resume_origin(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
    ) -> Result<(), Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        session
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::Other(Box::new(error)))?;
        if !runtime
            .backend()
            .memory_pool()
            .same_domain(self.sampling.arrays.custody.pool())
        {
            return Err(mismatch());
        }
        session
            .payload
            .model
            .erased()
            .validate_resident_control_origin(&self.decoder.origin)?;
        let mut epoch = Some(self.sampling.parameter_epoch.ok_or_else(mismatch)?);
        session.validate_parameter_epoch(&mut epoch)?;
        Ok(())
    }

    /// The exact immutable cache carried by this saved decoder, without lending
    /// mutable state, a request or any previously issued execution authority.
    pub(in crate::composition::mlx::session) fn input_identity(
        &self,
    ) -> Option<&SharedPreparedInputCacheIdentity> {
        self.decoder.input.as_ref()
    }

    pub(in crate::composition::mlx::session) fn sampling(&self) -> &CopiedTextSampling {
        &self.sampling
    }

    pub(in crate::composition::mlx::session) fn bytes(&self) -> u64 {
        self.sampling.bytes()
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
