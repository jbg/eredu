//! Indexed Candle placement and native execution context.

use std::{
    any::Any,
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use candle_core::Device;
use eredu_core::{AutomaticPlanningError, BackendFailure, BackendFailureKind, DevicePlan};

/// An exact Candle device selection that can be compared without opening a queue.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CandlePlacement {
    /// Host execution.
    #[default]
    Cpu,
    /// Metal device at the specified native index.
    Metal(usize),
    /// CUDA device at the specified native index.
    Cuda(usize),
}

impl CandlePlacement {
    /// Validates the canonical device identity and compiled feature support.
    pub fn from_plan(plan: &DevicePlan) -> Result<Self, AutomaticPlanningError> {
        let (family, index) = eredu_core::automatic::parse_indexed_device(plan, "candle")?;
        match family {
            "cpu" if index == 0 => Ok(Self::Cpu),
            "metal" if cfg!(all(feature = "metal", target_os = "macos")) => Ok(Self::Metal(index)),
            "cuda" if cfg!(feature = "cuda") && i32::try_from(index).is_ok() => {
                Ok(Self::Cuda(index))
            }
            _ => Err(AutomaticPlanningError::Invalid(format!(
                "Candle does not support the requested placement {family}:{index} in this build"
            ))),
        }
    }

    /// Returns the canonical device identity for this placement.
    pub fn device_plan(self) -> DevicePlan {
        let (family, index) = self.family_index();
        DevicePlan::new("candle", format!("{family}:{index}"))
            .expect("native Candle placement has a canonical identity")
    }

    pub(crate) const fn family_index(self) -> (&'static str, usize) {
        match self {
            Self::Cpu => ("cpu", 0),
            Self::Metal(index) => ("metal", index),
            Self::Cuda(index) => ("cuda", index),
        }
    }

    pub(crate) const fn is_metal(self) -> bool {
        matches!(self, Self::Metal(_))
    }

    pub(crate) const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub(crate) const fn supports_word_matrix(self, rank: usize) -> bool {
        rank == 2 || (rank == 3 && self.supports_local_text_extensions())
    }

    pub(crate) const fn supports_local_text_extensions(self) -> bool {
        self.is_cpu() || self.is_metal()
    }
}

/// The native device used for allocations and execution.
#[derive(Clone, Debug)]
pub struct CandleContext {
    pub(crate) device: Device,
    placement: CandlePlacement,
    fenced: Arc<AtomicBool>,
}

impl CandleContext {
    /// Creates a context for synchronous host execution.
    pub fn cpu() -> Self {
        Self::from_device(Device::Cpu, CandlePlacement::Cpu)
    }

    /// Opens the exact requested native device.
    pub fn open(plan: &DevicePlan) -> Result<Self, BackendFailure> {
        let placement = CandlePlacement::from_plan(plan)
            .map_err(|source| BackendFailure::new(BackendFailureKind::InvalidInput, source))?;
        let device = match placement {
            CandlePlacement::Cpu => Ok(Device::Cpu),
            CandlePlacement::Metal(index) => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if crate::memory::metal_device(index).is_none() {
                    return Err(crate::text::invalid("Metal device index is unavailable"));
                }
                Device::new_metal(index)
            }
            CandlePlacement::Cuda(index) => Device::new_cuda_with_stream(index),
        }
        .map_err(BackendFailure::from_error)?;
        Ok(Self::from_device(device, placement))
    }

    fn from_device(device: Device, placement: CandlePlacement) -> Self {
        Self {
            device,
            placement,
            fenced: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns the exact realized device placement.
    pub fn placement(&self) -> CandlePlacement {
        self.placement
    }

    pub(crate) fn same_context(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.fenced, &other.fenced)
    }

    pub(crate) fn fenced(&self) -> bool {
        self.fenced.load(Ordering::Acquire)
    }

    pub(crate) fn fence(&self) {
        self.fenced.store(true, Ordering::Release);
    }

    pub(crate) fn check_healthy(&self) -> Result<(), BackendFailure> {
        if self.fenced() {
            return Err(BackendFailure::new(
                BackendFailureKind::InvalidSession,
                std::io::Error::other("Candle context is fenced after a synchronization failure"),
            ));
        }
        Ok(())
    }

    pub(crate) fn synchronize(&self) -> Result<(), BackendFailure> {
        Self::synchronize_device(&self.device).map_err(|error| {
            self.fence();
            BackendFailure::from_error(error).with_operation("synchronize Candle device")
        })
    }

    pub(crate) fn synchronize_device(device: &Device) -> candle_core::Result<()> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if device.is_metal() {
            // Candle takes the in-flight command buffers out of its queue before
            // waiting. A second waiter must not mistake that for an idle queue.
            static METAL_WAIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _wait = METAL_WAIT
                .lock()
                .map_err(|_| candle_core::Error::Msg("Candle Metal wait lock poisoned".into()))?;
            return device.synchronize();
        }
        device.synchronize()
    }

    pub(crate) fn validate_tensor_device(
        &self,
        tensor: &crate::CandleTensor,
    ) -> Result<(), BackendFailure> {
        if !self.device.same_device(tensor.value.device()) {
            return Err(crate::text::invalid(
                "tensor belongs to a different Candle context",
            ));
        }
        Ok(())
    }
}

struct Unsettled {
    device: Device,
    resources: Option<Box<dyn Any>>,
}

impl Drop for Unsettled {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            if CandleContext::synchronize_device(&self.device).is_err() {
                // Preserve native roots and submission authority if retirement cannot be proven.
                std::mem::forget((self.device.clone(), resources));
            }
        }
    }
}

thread_local! {
    static UNSETTLED: RefCell<Vec<Unsettled>> = const { RefCell::new(Vec::new()) };
}

impl CandleContext {
    pub(crate) fn settle_roots<'a>(
        roots: impl IntoIterator<Item = &'a crate::CandleTensor>,
    ) -> Result<(), BackendFailure> {
        let mut devices: Vec<(&Device, Vec<&crate::CandleTensor>)> = Vec::new();
        for root in roots {
            let device = root.value.device();
            if device.is_cpu() {
                continue;
            }
            if let Some((_, roots)) = devices
                .iter_mut()
                .find(|(other, _)| device.same_device(other))
            {
                roots.push(root);
            } else {
                devices.push((device, vec![root]));
            }
        }
        let mut settled = Ok(());
        for (device, roots) in devices {
            if let Err(error) = Self::synchronize_device(device) {
                Self::quarantine_device(device, roots.into_iter().cloned().collect::<Vec<_>>());
                settled = settled.and(Err(BackendFailure::from_error(error)));
            }
        }
        settled
    }

    pub(crate) fn quarantine(&self, resources: impl Any) {
        Self::quarantine_device(&self.device, resources);
    }

    pub(crate) fn quarantine_device(device: &Device, resources: impl Any) {
        let mut work = Some(Unsettled {
            device: device.clone(),
            resources: Some(Box::new(resources)),
        });
        let _ = UNSETTLED.try_with(|pending| {
            if let Ok(mut pending) = pending.try_borrow_mut() {
                pending.extend(work.take());
            }
        });
        // During thread teardown a late guard may outlive the retirement list.
        // Its Unsettled owner then attempts retirement directly.
    }

    pub(crate) fn reclaim_retired() {
        let mut submissions = UNSETTLED.with(|pending| std::mem::take(&mut *pending.borrow_mut()));
        submissions.retain_mut(|work| {
            if Self::synchronize_device(&work.device).is_err() {
                return true;
            }
            work.resources.take();
            false
        });
        UNSETTLED.with(|pending| pending.borrow_mut().append(&mut submissions));
    }
}

/// Keeps resources and their authority alive until the configured device settles.
pub(crate) struct CandleSettlement<R: 'static> {
    pub(crate) queue: CandleContext,
    resources: Option<R>,
}

impl<R: 'static> CandleSettlement<R> {
    pub(crate) fn new(queue: &CandleContext, resources: R) -> Self {
        Self {
            queue: queue.clone(),
            resources: Some(resources),
        }
    }

    pub(crate) fn resources(&self) -> &R {
        self.resources.as_ref().expect("live native scope")
    }

    pub(crate) fn resources_mut(&mut self) -> &mut R {
        self.resources.as_mut().expect("live native scope")
    }

    pub(crate) fn settle(&mut self) -> Result<(), BackendFailure> {
        if let Err(error) = self.queue.synchronize() {
            if let Some(resources) = self.resources.take() {
                self.queue.quarantine(resources);
            }
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<R, BackendFailure> {
        self.settle()?;
        Ok(self.resources.take().expect("settled native scope"))
    }

    pub(crate) fn release_settled(mut self) {
        self.resources.take();
    }
}

impl<R: 'static> Drop for CandleSettlement<R> {
    fn drop(&mut self) {
        if self.resources.is_some() {
            let _ = self.settle();
        }
    }
}
