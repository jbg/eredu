//! Loaded parameter operations for resident and partitioned text sessions.

use super::InstalledTextParameterOverlay;
use crate::executable_binding::ErasedTextExecutable as _;
use eredu_core::parameters::native::{ParameterNative, ParameterReadFragment};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use eredu_core::{
    capture::{add, mul, CaptureError, CaptureReservation, CaptureSkipReason, CaptureUsage},
    execution_control::SnapshotEstimate,
    intervention::InterventionDtype,
    parameters::*,
    TextGenerationBackend,
};

use crate::parameter_operations::*;

use super::{TextExecutable, TextSessionDriver, TextSessionMechanisms};

mod catalog;
mod effective;
use effective::EffectiveLayout;
mod loans;
mod local;
mod overlay;
mod read;

pub use catalog::ParameterSlotLayout;
use catalog::{partition_catalog, PartitionedParameters};
use overlay::{partition_activation, partition_removal};
pub use overlay::{LocalParameterEdit, LocalParameterTarget};
use read::partition_read;

type Session<B, N> = TextSessionDriver<<N as TextSessionMechanisms<B>>::Resources, N>;
type Model<B, N> = <N as TextSessionMechanisms<B>>::Model;
type Context<B, N> = <<N as TextSessionMechanisms<B>>::Executable as TextExecutable<
    crate::text_session_driver::SessionTextTypes<B, N>,
>>::Context;

type ParameterTensors<B, N> =
    BTreeMap<String, <N as eredu_core::backend::TextSessionNative<B>>::ParameterValue>;

/// Native parameter access composed with the installed portable text session.
trait ParameterMechanisms<B: TextGenerationBackend>:
    ParameterNative<B> + TextSessionMechanisms<B>
{
}

impl<B, N> ParameterMechanisms<B> for N
where
    B: TextGenerationBackend,
    N: ParameterNative<B> + TextSessionMechanisms<B>,
{
}

const UNLIMITED: CaptureUsage = CaptureUsage {
    captures: u64::MAX,
    retained_bytes: u64::MAX,
    host_bytes: u64::MAX,
    encoded_bytes: u64::MAX,
};
impl<B, N> ParameterSession<B> for Session<B, N>
where
    B: TextGenerationBackend,
    N: ParameterMechanisms<B>,
    B::Error: Send + Sync + 'static,
{
    fn parameter_discovery(&mut self, _: &B) -> Result<ParameterDiscovery, ParameterError> {
        match N::control_transport(&self.resources().retained) {
            Some(_) => partition_catalog::<B, N>(self, None).map(|catalog| catalog.discovery),
            None => local::local_catalog::<B, N>(self).map(|(discovery, _)| discovery),
        }
    }

    fn query_parameter(
        &mut self,
        backend: &B,
        identity: &str,
        parameter: &str,
        region: ParameterRegion,
        limits: CaptureUsage,
    ) -> Result<ParameterValues, ParameterError> {
        let read = if N::control_transport(&self.resources().retained).is_none() {
            local::local_read::<B, N>
        } else {
            partition_read::<B, N>
        };
        let (dtype, _, values) = read(
            self,
            identity,
            parameter,
            &region,
            None,
            limits,
            N::context(backend),
        )?;
        Ok(ParameterValues {
            identity: identity.into(),
            parameter: parameter.into(),
            dtype,
            region,
            values,
            usage: self.parameter_usage.get(),
        })
    }

    fn project_parameter(
        &mut self,
        backend: &B,
        identity: &str,
        parameter: &str,
        projection: ParameterProjection,
        limits: CaptureUsage,
    ) -> Result<ParameterProjectionValues, ParameterError> {
        let read = if N::control_transport(&self.resources().retained).is_none() {
            local::local_read::<B, N>
        } else {
            partition_read::<B, N>
        };
        let (source_dtype, shape, values) = read(
            self,
            identity,
            parameter,
            &projection.region,
            Some(&projection),
            limits,
            N::context(backend),
        )?;
        Ok(ParameterProjectionValues {
            identity: identity.into(),
            parameter: parameter.into(),
            source_dtype,
            shape,
            values,
            usage: self.parameter_usage.get(),
        })
    }

    fn activate_parameter_overlay(
        &mut self,
        backend: &B,
        overlay: &AdmittedParameterOverlay,
        limits: CaptureUsage,
    ) -> Result<ParameterDiscovery, ParameterError> {
        if N::control_transport(&self.resources().retained).is_none() {
            return local::local_activation::<B, N>(self, overlay, limits, N::context(backend));
        }
        partition_activation::<B, N>(self, overlay, limits, N::context(backend))
    }

    fn remove_parameter_overlay(
        &mut self,
        _: &B,
        identity: &str,
    ) -> Result<ParameterDiscovery, ParameterError> {
        if N::control_transport(&self.resources().retained).is_none() {
            return local::local_removal::<B, N>(self, identity);
        }
        partition_removal::<B, N>(self, identity)
    }
}

fn finish_control<B, N, R>(
    session: &Session<B, N>,
    transport: &N::ControlTransport,
    result: Result<R, ParameterError>,
) -> Result<R, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    if result.is_err() && transport.ensure_parameter_active().is_err() {
        session.native().fence_parameter_session();
    }
    result
}

fn parameter_failure(error: impl std::error::Error + Send + Sync + 'static) -> ParameterError {
    eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::Other, error).into()
}

/// An independently borrowed counter permits charging metadata, native loans
/// and delivery without retaining or mutably borrowing the model payload.
#[derive(Clone)]
struct ParameterBudget {
    total: Rc<Cell<CaptureUsage>>,
    limit: CaptureUsage,
    operation: Option<(Rc<Cell<CaptureUsage>>, CaptureUsage)>,
}
impl CaptureReservation for ParameterBudget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        let next = self.total.get().checked_add(cost)?;
        if let Some(budget) = next.exceeded(self.limit) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: true,
            });
        }
        if let Some((used, limit)) = &self.operation {
            let next_operation = used.get().checked_add(cost)?;
            if let Some(budget) = next_operation.exceeded(*limit) {
                return Err(CaptureError::Limit {
                    budget,
                    cumulative: false,
                });
            }
            used.set(next_operation);
        }
        self.total.set(next);
        Ok(None)
    }
}

pub(super) struct ActiveParameterOverlay {
    identity: String,
    precision: Option<std::num::NonZeroU8>,
    baseline_transforms: BTreeMap<String, ProjectionInputTransform>,
}

fn parameter_width(dtype: Option<InterventionDtype>) -> Option<std::num::NonZeroU8> {
    std::num::NonZeroU8::new(match dtype {
        Some(InterventionDtype::Float32) => 4,
        Some(InterventionDtype::Float16 | InterventionDtype::Bfloat16) => 2,
        None => 0,
    })
}

impl<R, N> TextSessionDriver<R, N> {
    /// Identity of the currently installed overlay for capture and intervention admission.
    pub fn parameter_overlay_identity(&self) -> Option<&str> {
        self.parameter_overlay
            .as_ref()
            .map(|overlay| overlay.identity.as_str())
    }

    /// Checks whether the installed parameters can use persisted prompt cache state.
    pub fn ensure_parameter_cache_compatible(
        &self,
    ) -> Result<(), eredu_core::cache::PromptCacheError> {
        if self.parameter_overlay.is_some() {
            return Err(eredu_core::cache::PromptCacheError::Incompatible(
                "persistent prompt caches do not yet encode active parameter overlays; use a fresh exact-token prefill".into(),
            ));
        }
        Ok(())
    }

    /// Upper bound for floating state precision including promotion by edited parameters.
    pub fn effective_floating_state_dtype_bytes(&self) -> std::num::NonZeroU8 {
        let baseline = self.layout.floating_state_dtype_bytes();
        self.parameter_overlay
            .as_ref()
            .and_then(|overlay| overlay.precision)
            .map_or(baseline, |edited| edited.max(baseline))
    }

    fn commit_parameter_overlay(
        &mut self,
        discovery: &mut ParameterDiscovery,
        epoch: u64,
        next: Option<ActiveParameterOverlay>,
    ) {
        self.parameter_overlay = next;
        self.parameter_epoch = epoch;
        discovery.overlay_identity = self.parameter_overlay_identity().map(str::to_owned);
        discovery.identity = format!("{}:parameters:{epoch}", self.intervention_identity);
        discovery.usage = self.parameter_usage.get();
    }

    fn parameter_budget(
        &self,
        limit: CaptureUsage,
        operation_limit: Option<CaptureUsage>,
    ) -> ParameterBudget {
        ParameterBudget {
            total: Rc::clone(&self.parameter_usage),
            limit,
            operation: operation_limit
                .map(|limit| (Rc::new(Cell::new(CaptureUsage::default())), limit)),
        }
    }
}
