//! Actual current GGUF miss: one immutable copy per physical output.
mod source_publication;
mod supplied;
mod transform;
use super::super::leases::WeightLeaseSource;
use super::*;
use crate::backend::runtime::checkpoint::gguf::{self, producer::ArrayProducer};
use crate::backend::runtime::execution::generic::gguf_host_typed::{
    DestinationCause, TransformKind, TypedDestination, TypedOutputRequest, TypedTransformFacts,
    TypedValues,
};
use eredu_checkpoint::gguf_store::{GgufConversionPlan, GgufConversionPlanError};
use eredu_gguf::LogicalDtype;
use eredu_runtime::working_memory::{
    HostDestinationCause, OriginalHostDestinationBank, OriginalHostSourceBank,
    OriginalHostSourceError, OriginalHostSourceReceipt,
};
use safemlx::{
    error::IoError, OwnedHostBufferCopyError, OwnedHostCopyCause, OwnedHostCopyFacts,
    OwnedHostCopyPlan, OwnedHostCopyPreparationError, PreparedInputRuntime,
    PreparedSubmissionGraphQuota, SubmissionGraphQuotaCause,
};
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

/// Fixed physical-copy failure or the actual typed converter/native source.
#[derive(Debug)]
pub enum GgufHostCopyCause {
    /// Actual structural or typed conversion failure, without formatting it.
    Conversion(IoError),
    /// Exact safe-copy outcome and its immutable native source snapshot.
    Copy {
        /// Real fixed copy outcome; Busy is not terminal native failure.
        cause: OwnedHostCopyCause,
        /// Actual native source when the producer returned one.
        native: Option<safemlx::error::Exception>,
    },
    /// Actual source-metadata preparation refusal.
    Arena(SubmissionGraphQuotaCause),
    /// Reached source-arena debit refused before either allocation.
    SourceFunding(OriginalHostSourceError),
    /// Exact prepaid-source registry refusal; backing remains with recovery.
    SourcePublication(eredu_runtime::working_memory::WorkingMemoryError),
    /// Positive immutable inspection or checked attachment refused.
    SourceNative(safemlx::OriginalBufferCause),
    /// Actual cache metadata provider/extent failure; the attempted key stays
    /// in pending recovery with all initialized destination storage.
    CacheStorage(eredu_gguf::SuppliedStorageError<HostDestinationCause>),
    /// Actual fixed prepared-owner allocation failure, preserving its source.
    SourceOwner(safemlx::PreparedAllocationOwnerCause),
    /// The caller did not supply an ordinarily prepared runtime witness.
    MissingRuntime,
    /// Actual same-source selected plan error, preserving its neutral source.
    SourcePlan(GgufConversionPlanError),
    /// Actual failed reservation of the reached typed destination.
    TypedReservation(TryReserveError),
    /// Actual original host admission or no-growth storage failure.
    HostDestination(HostDestinationCause),
    /// The selected source/output cannot supply this reached typed transformation.
    TypedBinding,
    /// A bounded write refused an unexpected extent without growing its Vec.
    TypedExtent {
        /// Maximum initialized elements requested for this output.
        requested: usize,
        /// Observed attempted or supplied initialized elements.
        actual: usize,
    },
    /// Prepared destination capacity did not cover its requested writes.
    TypedCapacity {
        /// Elements the reached destination was asked to reserve.
        requested: usize,
        /// Capacity of the actual retained destination.
        actual: usize,
    },
    /// A once-only destination was already used.
    TypedAlreadyFilled,
    /// Checked concrete control layout could not be represented.
    Layout,
}
impl From<IoError> for GgufHostCopyCause {
    fn from(value: IoError) -> Self {
        Self::Conversion(value)
    }
}
impl fmt::Display for GgufHostCopyCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conversion(value) => value.fmt(f),
            Self::Copy { cause, .. } => cause.fmt(f),
            Self::Arena(cause) => cause.fmt(f),
            Self::SourceFunding(cause) => cause.fmt(f),
            Self::SourcePublication(cause) => cause.fmt(f),
            Self::SourceNative(cause) => cause.fmt(f),
            Self::SourceOwner(cause) => cause.fmt(f),
            Self::CacheStorage(cause) => cause.fmt(f),
            Self::SourcePlan(cause) => cause.fmt(f),
            Self::HostDestination(cause) => cause.fmt(f),
            Self::TypedReservation(cause) => cause.fmt(f),
            Self::TypedBinding => f.write_str("original GGUF typed source/output binding mismatch"),
            Self::TypedExtent { requested, actual } => write!(
                f,
                "original GGUF typed extent {actual} exceeds request {requested}"
            ),
            Self::TypedCapacity { requested, actual } => write!(
                f,
                "original GGUF typed capacity {actual} is below request {requested}"
            ),
            Self::TypedAlreadyFilled => {
                f.write_str("original GGUF typed destination was already filled")
            }
            Self::Layout => f.write_str("original GGUF host-copy layout overflow"),
            Self::MissingRuntime => f.write_str("original GGUF host-copy runtime was not prepared"),
        }
    }
}
impl std::error::Error for GgufHostCopyCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conversion(value) => Some(value),
            Self::Copy {
                native: Some(value),
                ..
            } => Some(value),
            Self::Copy { cause, .. } => Some(cause),
            Self::Arena(value) => Some(value),
            Self::SourceFunding(value) => Some(value),
            Self::SourcePublication(value) => Some(value),
            Self::SourceNative(value) => Some(value),
            Self::SourceOwner(value) => Some(value),
            Self::CacheStorage(value) => Some(value),
            Self::SourcePlan(cause) => Some(cause),
            Self::HostDestination(cause) => Some(cause),
            Self::TypedReservation(cause) => Some(cause),
            Self::MissingRuntime
            | Self::Layout
            | Self::TypedBinding
            | Self::TypedExtent { .. }
            | Self::TypedCapacity { .. }
            | Self::TypedAlreadyFilled => None,
        }
    }
}

impl From<DestinationCause> for GgufHostCopyCause {
    fn from(cause: DestinationCause) -> Self {
        match cause {
            DestinationCause::Layout => Self::Layout,
            DestinationCause::Original(cause) => Self::HostDestination(cause),
            DestinationCause::Reserve(cause) => Self::TypedReservation(cause),
            DestinationCause::Extent { requested, actual } => {
                Self::TypedExtent { requested, actual }
            }
            DestinationCause::Capacity { requested, actual } => {
                Self::TypedCapacity { requested, actual }
            }
            DestinationCause::AlreadyFilled => Self::TypedAlreadyFilled,
        }
    }
}

enum TransformInput {
    Bytes(TypedValues<u8>),
    HalfBits(TypedValues<u16>),
}

// Empty only before the same neutral Box is installed, synchronously and before
// any native call. The shared value never owns a manager or a native object.
#[derive(Debug)]
pub(in crate::backend::runtime::checkpoint::store) struct OriginalGgufSource {
    lease: Option<Box<NeutralGgufLease>>,
}
impl OriginalGgufSource {
    pub(in crate::backend::runtime::checkpoint::store) fn lease(&self) -> &NeutralGgufLease {
        self.lease.as_deref().expect("bound original GGUF source")
    }
}
#[derive(Clone, Debug)]
pub(in crate::backend::runtime::checkpoint::store) struct SourceCustody {
    source: Arc<OriginalGgufSource>,
    // Outside the Arc: its allocation and actual source Box retire first.
    controls: OriginalTextControlGuard,
    acquisition_metadata: Option<eredu_runtime::working_memory::OriginalHostMetadataCustody>,
}

// Unlike lease custody, this owner cannot be cloned. The existing native arena
// frees itself and its Rust queue shell before this final accounting receipt.
#[derive(Debug)]
struct SourceArenaCustody {
    source: SourceCustody,
    receipt: Option<OriginalHostSourceReceipt>,
}

impl SourceCustody {
    pub(in crate::backend::runtime::checkpoint::store) fn lease(&self) -> &NeutralGgufLease {
        self.source.lease()
    }
}

/// Compact Send+Sync transport. The thread-affine failed copy owner stays in
/// the already allocated pending recovery node, never in this public error.
#[derive(Debug)]
pub struct PreparedGgufHostCopyFailure {
    cause: Box<GgufHostCopyCause>,
    output: usize,
    custody: SourceCustody,
}
impl PreparedGgufHostCopyFailure {
    /// Actual fixed/converter/native cause, preserving its typed source chain.
    pub fn cause(&self) -> &GgufHostCopyCause {
        &self.cause
    }
    /// Physical output ordinal in the shared converter's original order.
    pub fn output_ordinal(&self) -> usize {
        self.output
    }
    fn control_bytes() -> Option<usize> {
        size_of::<Self>()
            .checked_add(size_of::<GgufHostCopyCause>())?
            .checked_add(size_of::<Result<gguf::GgufTensor, Self>>())?
            .checked_add(size_of::<Result<CachedGgufArrays, Self>>())
    }
}
impl fmt::Display for PreparedGgufHostCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for PreparedGgufHostCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}

// An unfilled Vec and any real partial arena, or the actual safe wrapper's
// owning failed attempt. Nothing is inferred from a polling outcome.
enum TypedFailure<T: safemlx::ArrayElement + Send + 'static> {
    Transform {
        input: TransformInput,
        destination: Option<TypedDestination<T>>,
    },
    Input {
        values: TypedValues<T>,
        preparation: Option<OwnedHostCopyPreparationError<SourceArenaCustody>>,
    },
    Copy(OwnedHostBufferCopyError<T, SourceArenaCustody, TypedValues<T>>),
    Publication(source_publication::Failure<T>),
}
enum FailedCopy {
    F32(TypedFailure<f32>),
    F16(TypedFailure<half::f16>),
    Bf16(TypedFailure<half::bf16>),
    I8(TypedFailure<i8>),
    I16(TypedFailure<i16>),
    I32(TypedFailure<i32>),
    I64(TypedFailure<i64>),
    F64(TypedFailure<f64>),
    U8(TypedFailure<u8>),
    U32(TypedFailure<u32>),
}

pub(super) struct PreparedGgufHostCopy {
    // Failed Vec/slot/arena before source custody and its final guard.
    failed: Option<FailedCopy>,
    cache: Option<super::super::cache::PreparedEntry>,
    pub(super) allocation_source: Option<usize>,
    pub(super) raw_attempted: bool,
    pub(super) raw_failure: Option<PreparedGgufAdmittedFailure>,
    pub(super) host_destinations: Option<OriginalHostDestinationBank>,
    pub(super) source_constructions: Option<OriginalHostSourceBank>,
    #[cfg(test)]
    copy_observer: Option<(usize, OriginalScopeObserver)>,
    #[cfg(test)]
    last_input: Option<(usize, usize, usize)>,
    source_plan: Option<Result<GgufConversionPlan, GgufConversionPlanError>>,
    source_plan_matches: bool,
    transforms: [Option<TypedTransformFacts>; 3],
    runtime: Option<Rc<PreparedInputRuntime>>,
    facts: [Option<OwnedHostCopyFacts>; 3],
    copy_controls: [Option<usize>; 3],
    produced: usize,
    source: Arc<OriginalGgufSource>,
    controls: OriginalTextControlGuard,
    acquisition_metadata: Option<eredu_runtime::working_memory::OriginalHostMetadataCustody>,
}
impl PreparedGgufHostCopy {
    #[cfg(test)]
    pub(super) fn retained_cache_group(&self) -> Option<&super::super::cache::Group> {
        self.cache
            .as_ref()
            .and_then(super::super::cache::PreparedEntry::retained_group)
    }
    pub(super) fn new(
        runtime: Option<Rc<PreparedInputRuntime>>,
        controls: OriginalTextControlGuard,
    ) -> Self {
        Self {
            failed: None,
            cache: None,
            allocation_source: None,
            raw_attempted: false,
            raw_failure: None,
            host_destinations: None,
            source_constructions: None,
            #[cfg(test)]
            copy_observer: None,
            #[cfg(test)]
            last_input: None,
            source_plan: None,
            source_plan_matches: false,
            transforms: [None; 3],
            runtime,
            facts: [None; 3],
            copy_controls: [None; 3],
            produced: 0,
            source: Arc::new(OriginalGgufSource { lease: None }),
            controls,
            acquisition_metadata: None,
        }
    }
    pub(super) fn control_bytes() -> Option<u64> {
        // Actual Rust 1.98 ArcInner representation, matching the existing
        // request-bank source policy; allocator rounding remains pending.
        let arc = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(Layout::new::<OriginalGgufSource>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let bytes = arc
            .checked_add(PreparedGgufHostCopyFailure::control_bytes()?)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<SourceCustody>())?
            .checked_add(size_of::<safemlx::OriginalScopeObserver>())?;
        u64::try_from(bytes).ok()
    }
    fn custody(&self) -> SourceCustody {
        SourceCustody {
            source: Arc::clone(&self.source),
            controls: self.controls.clone(),
            acquisition_metadata: self.acquisition_metadata.clone(),
        }
    }
    fn failure(&self, cause: GgufHostCopyCause) -> PreparedGgufHostCopyFailure {
        PreparedGgufHostCopyFailure {
            // The real source/control owners already survive this Box allocation.
            cause: Box::new(cause),
            output: self.produced,
            custody: self.custody(),
        }
    }
    fn create<T: safemlx::ArrayElement + Send + 'static>(
        &mut self,
        values: TypedValues<T>,
        shape: &[i32],
        dtype: LogicalDtype,
        observer: &OriginalScopeObserver,
        retain: fn(TypedFailure<T>) -> FailedCopy,
    ) -> Result<Array, GgufHostCopyCause> {
        assert!(
            self.failed.is_none(),
            "one physical-copy failure per pending owner"
        );
        assert!(
            self.produced < self.facts.len(),
            "closed GGUF output population"
        );
        #[cfg(test)]
        {
            self.last_input = Some((values.as_ptr() as usize, values.len(), values.capacity()));
        }
        let request = match self.output_request(dtype) {
            Ok(request) => request,
            Err(cause) => {
                self.failed = Some(retain(TypedFailure::Input {
                    values,
                    preparation: None,
                }));
                return Err(cause);
            }
        };
        if values.len() > request.output_elements {
            let cause = GgufHostCopyCause::TypedExtent {
                requested: request.output_elements,
                actual: values.len(),
            };
            self.failed = Some(retain(TypedFailure::Input {
                values,
                preparation: None,
            }));
            return Err(cause);
        }
        if request.kind == TransformKind::Move {
            self.transforms[self.produced] = Some(TypedTransformFacts {
                input_elements: values.len(),
                input_capacity: values.capacity(),
                requested_output_elements: request.output_elements,
                output_elements: values.len(),
                output_capacity: values.capacity(),
                input_element_bytes: size_of::<T>(),
                output_element_bytes: size_of::<T>(),
                separate_destination: false,
            });
        }
        let Some(runtime) = self.runtime.as_ref() else {
            self.failed = Some(retain(TypedFailure::Input {
                values,
                preparation: None,
            }));
            return Err(GgufHostCopyCause::MissingRuntime);
        };
        let plan = match OwnedHostCopyPlan::<T>::new(
            runtime,
            shape,
            values
                .admitted_capacity()
                .unwrap_or_else(|| values.capacity()),
        ) {
            Ok(plan) => plan,
            Err(cause) => {
                self.failed = Some(retain(TypedFailure::Input {
                    values,
                    preparation: None,
                }));
                return Err(GgufHostCopyCause::Copy {
                    cause,
                    native: None,
                });
            }
        };
        let storage = match SourceCopyStorage::for_plan(&plan, self.source_constructions.is_some())
        {
            Ok(storage) => storage,
            Err(cause) => {
                self.failed = Some(retain(TypedFailure::Input {
                    values,
                    preparation: None,
                }));
                return Err(cause);
            }
        };
        let facts = plan.facts();
        self.copy_controls[self.produced] = Some(storage.controls);
        self.facts[self.produced] = Some(facts);
        let custody = self.custody();
        #[cfg(test)]
        let observer = self
            .copy_observer
            .as_ref()
            .filter(|(ordinal, _)| *ordinal == self.produced)
            .map_or(observer, |(_, observer)| observer);
        if let Some(bank) = self.source_constructions.as_mut() {
            match source_publication::create(bank, plan, values, custody, storage, observer) {
                Ok(array) => {
                    self.produced += 1;
                    return Ok(array);
                }
                Err((cause, owner)) => {
                    self.failed = Some(retain(owner));
                    return Err(cause);
                }
            }
        }
        // The ordinary unbanked adapter preserves its existing source lifetime;
        // it cannot produce a prepaid canonical source origin.
        let receipt = None;
        let custody = SourceArenaCustody {
            source: custody,
            receipt,
        };
        let arena = match PreparedSubmissionGraphQuota::try_new(facts.metadata_bytes(), custody) {
            Ok(arena) => arena,
            Err(error) => {
                let (cause, custody) = error.into_parts();
                self.failed = Some(retain(TypedFailure::Input {
                    values,
                    preparation: None,
                }));
                drop(custody); // the same source also remains in this pending node
                return Err(GgufHostCopyCause::Arena(cause));
            }
        };
        let ready = match plan.prepare(arena) {
            Ok(ready) => ready,
            Err(error) => {
                let cause = error.cause();
                self.failed = Some(retain(TypedFailure::Input {
                    values,
                    preparation: Some(error),
                }));
                return Err(GgufHostCopyCause::Copy {
                    cause,
                    native: None,
                });
            }
        };
        match ready.try_fill_owned(values, observer) {
            Ok(array) => {
                self.produced += 1;
                Ok(array)
            }
            Err(mut error) => {
                let cause = error.cause();
                let native = error.take_native_source();
                // Move only the immutable diagnostic. The same Vec/slot/arena
                // remains pending; failure() adds independent source custody
                // before this diagnostic can escape through the public error.
                self.failed = Some(retain(TypedFailure::Copy(error)));
                Err(GgufHostCopyCause::Copy { cause, native })
            }
        }
    }
}
struct OriginalProducer<'a> {
    host: &'a mut PreparedGgufHostCopy,
    observer: &'a OriginalScopeObserver,
}
impl ArrayProducer for OriginalProducer<'_> {
    type Error = GgufHostCopyCause;
    fn bind_group(&mut self, descriptor: &eredu_gguf::TensorDescriptor) {
        self.host.bind_plan(descriptor);
    }
    fn f32_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::F32,
            f32::from_ne_bytes,
            FailedCopy::F32,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F32,
            self.observer,
            FailedCopy::F32,
        )
    }
    fn f16_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::F16,
            |bytes| half::f16::from_bits(u16::from_ne_bytes(bytes)),
            FailedCopy::F16,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F16,
            self.observer,
            FailedCopy::F16,
        )
    }
    fn bf16_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::Bf16,
            |bytes| half::bf16::from_bits(u16::from_ne_bytes(bytes)),
            FailedCopy::Bf16,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::Bf16,
            self.observer,
            FailedCopy::Bf16,
        )
    }
    fn i8_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I8,
            i8::from_ne_bytes,
            FailedCopy::I8,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I8,
            self.observer,
            FailedCopy::I8,
        )
    }
    fn i16_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I16,
            i16::from_ne_bytes,
            FailedCopy::I16,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I16,
            self.observer,
            FailedCopy::I16,
        )
    }
    fn i32_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I32,
            i32::from_ne_bytes,
            FailedCopy::I32,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I32,
            self.observer,
            FailedCopy::I32,
        )
    }
    fn i64_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I64,
            i64::from_ne_bytes,
            FailedCopy::I64,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I64,
            self.observer,
            FailedCopy::I64,
        )
    }
    fn f64_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::F64,
            f64::from_ne_bytes,
            FailedCopy::F64,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F64,
            self.observer,
            FailedCopy::F64,
        )
    }
    fn f16_bits(&mut self, bits: Vec<u16>, shape: &[i32]) -> Result<Array, Self::Error> {
        let values = self.host.transform_half_bits(bits)?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F16,
            self.observer,
            FailedCopy::F16,
        )
    }
    fn f32(&mut self, values: Vec<f32>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::F32,
            self.observer,
            FailedCopy::F32,
        )
    }
    fn f16(&mut self, values: Vec<half::f16>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::F16,
            self.observer,
            FailedCopy::F16,
        )
    }
    fn bf16(&mut self, values: Vec<half::bf16>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::Bf16,
            self.observer,
            FailedCopy::Bf16,
        )
    }
    fn i8(&mut self, values: Vec<i8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::I8,
            self.observer,
            FailedCopy::I8,
        )
    }
    fn i16(&mut self, values: Vec<i16>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::I16,
            self.observer,
            FailedCopy::I16,
        )
    }
    fn i32(&mut self, values: Vec<i32>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::I32,
            self.observer,
            FailedCopy::I32,
        )
    }
    fn i64(&mut self, values: Vec<i64>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::I64,
            self.observer,
            FailedCopy::I64,
        )
    }
    fn f64(&mut self, values: Vec<f64>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::F64,
            self.observer,
            FailedCopy::F64,
        )
    }
    fn u8(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::U8,
            self.observer,
            FailedCopy::U8,
        )
    }
    fn u32(&mut self, values: Vec<u32>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::U32,
            self.observer,
            FailedCopy::U32,
        )
    }
}
impl PendingWeightMaterialization {
    pub(in crate::backend::runtime::checkpoint::store) fn convert_original_gguf(
        &mut self,
        portable: eredu_gguf::ConvertedCheckpointTensor,
    ) -> Result<gguf::GgufTensor, PreparedGgufHostCopyFailure> {
        // Intrusive alias only; no child, Box or TLS-based mechanism choice.
        let observer = self
            .retained
            .original_observer()
            .expect("original pending recovery")
            .clone();
        let host = self.bind_original_gguf_host();
        let result = gguf::convert_tensor_with(
            portable,
            &mut OriginalProducer {
                host: &mut *host,
                observer: &observer,
            },
        );
        result.map_err(|cause| host.failure(cause))
    }
    pub(super) fn bind_original_gguf_host(&mut self) -> &mut PreparedGgufHostCopy {
        let pending = self.retained.retention_mut();
        let host = pending
            .gguf_host
            .as_mut()
            .expect("original prepared host-copy slot");
        let WeightLeaseSource::Gguf(source) = &mut pending.lease.source else {
            unreachable!("closed original GGUF conversion")
        };
        // The G1/G3 same-Box transaction has finished. Establish all source
        // aliases before any native copy or before an escaping public error.
        let original = Arc::get_mut(&mut host.source).expect("unissued original source holder");
        assert!(original.lease.is_none());
        original.lease = Some(source.lease.take());
        host.acquisition_metadata = pending.acquisition_metadata.clone();
        source.lease.bind_original(host.custody());
        host
    }
}

#[cfg(test)]
impl PreparedGgufHostCopy {
    pub(super) fn fail_using_observer(&mut self, output: usize, observer: &OriginalScopeObserver) {
        self.copy_observer = Some((output, observer.clone()));
    }
    pub(super) fn facts(&self) -> &[Option<OwnedHostCopyFacts>; 3] {
        &self.facts
    }
    pub(super) fn transform_facts(&self) -> &[Option<TypedTransformFacts>; 3] {
        &self.transforms
    }
    pub(super) fn has_source_plan(&self) -> bool {
        self.source_plan.is_some()
    }
    pub(super) fn test_transform_refusal(
        &mut self,
        descriptor: &eredu_gguf::TensorDescriptor,
        bytes: Vec<u8>,
        wrong_binding: bool,
    ) -> PreparedGgufHostCopyFailure {
        // Keep the genuine source/lease plan, but present a different physical
        // descriptor. Malformed width must still win when both are wrong.
        let mut different_descriptor = descriptor.clone();
        different_descriptor.data_offset = descriptor.data_offset.checked_add(1).unwrap();
        self.bind_plan(&different_descriptor);
        let plan = self.source_plan.as_ref().unwrap().as_ref().unwrap();
        assert!(plan.matches_lease(self.source.lease()));
        assert!(!self.source_plan_matches);
        let identity = (bytes.as_ptr(), bytes.len(), bytes.capacity());
        let cause = self
            .transform_bytes(
                bytes,
                LogicalDtype::F32,
                f32::from_ne_bytes,
                FailedCopy::F32,
            )
            .expect_err("fixture transform refusal");
        let Some(FailedCopy::F32(TypedFailure::Transform {
            input: TransformInput::Bytes(input),
            destination,
        })) = self.failed.as_ref()
        else {
            panic!("same pending owner must retain the failed encoded input");
        };
        assert_eq!((input.as_ptr(), input.len(), input.capacity()), identity);
        assert!(
            destination.is_none(),
            "width and binding checks precede the destination request"
        );
        assert!(self.facts.iter().all(Option::is_none));
        assert!(self.transforms.iter().all(Option::is_none));
        if wrong_binding {
            assert!(matches!(&cause, GgufHostCopyCause::TypedBinding));
        } else {
            assert!(matches!(
                &cause,
                GgufHostCopyCause::Conversion(IoError::InvalidFormat(_))
            ));
        }
        self.failure(cause)
    }
    pub(super) fn assert_bound_source_plan(&self) {
        let plan = self
            .source_plan
            .as_ref()
            .expect("reached group plan")
            .as_ref()
            .unwrap();
        assert!(self.source_plan_matches);
        assert!(plan.matches_lease(self.source.lease()));
    }

    pub(super) fn source_owner(&self) -> std::sync::Weak<OriginalGgufSource> {
        Arc::downgrade(&self.source)
    }
    pub(super) fn assert_source_refusal_precedes_arena(&self) {
        fn before<T: safemlx::ArrayElement + Send + 'static>(failure: &TypedFailure<T>) {
            assert!(matches!(
                failure,
                TypedFailure::Input {
                    preparation: None,
                    ..
                }
            ));
        }
        match self.failed.as_ref().expect("retained input") {
            FailedCopy::F32(v) => before(v),
            FailedCopy::F16(v) => before(v),
            FailedCopy::Bf16(v) => before(v),
            FailedCopy::I8(v) => before(v),
            FailedCopy::I16(v) => before(v),
            FailedCopy::I32(v) => before(v),
            FailedCopy::I64(v) => before(v),
            FailedCopy::F64(v) => before(v),
            FailedCopy::U8(v) => before(v),
            FailedCopy::U32(v) => before(v),
        }
    }
    pub(super) fn assert_failed_input_unchanged(&self) {
        fn actual<T: safemlx::ArrayElement + Send + 'static>(
            failure: &TypedFailure<T>,
        ) -> (usize, usize, usize) {
            let values = match failure {
                TypedFailure::Input { values, .. } => values,
                TypedFailure::Copy(error) => error.buffer(),
                TypedFailure::Publication(error) => match error.cause() {
                    eredu_runtime::working_memory::OriginalHostSourceFailureCause::Native(
                        owner,
                    ) => owner.input(),
                    _ => panic!("expected entered copy failure"),
                },
                TypedFailure::Transform { .. } => {
                    panic!("expected native-copy failure, not transform refusal")
                }
            };
            (values.as_ptr() as usize, values.len(), values.capacity())
        }
        let shape = match self.failed.as_ref().expect("owning copy failure") {
            FailedCopy::F32(value) => actual(value),
            FailedCopy::F16(value) => actual(value),
            FailedCopy::Bf16(value) => actual(value),
            FailedCopy::I8(value) => actual(value),
            FailedCopy::I16(value) => actual(value),
            FailedCopy::I32(value) => actual(value),
            FailedCopy::I64(value) => actual(value),
            FailedCopy::F64(value) => actual(value),
            FailedCopy::U8(value) => actual(value),
            FailedCopy::U32(value) => actual(value),
        };
        assert_eq!(Some(shape), self.last_input);
    }
}

// The copied native source is separate from the admitted input Vec. These
// scalars describe its actual producer; they are not another budget or owner.
#[derive(Clone, Copy)]
struct SourceCopyStorage {
    controls: usize,
    backing: usize,
}
impl SourceCopyStorage {
    fn for_plan<T: safemlx::ArrayElement + Send + 'static>(
        plan: &OwnedHostCopyPlan<'_, T>,
        publish: bool,
    ) -> Result<Self, GgufHostCopyCause> {
        // Legacy unbanked copies do not prepare a canonical source row and must
        // not inherit the new registry's qualified-storage requirement.
        let publication = if publish {
            source_publication::control_bytes::<T>()
        } else {
            Some(0)
        };
        let controls = plan
            .control_bytes_with_buffer::<SourceArenaCustody, TypedValues<T>>()
            .and_then(|n| n.checked_add(publication?))
            // New reached/cold helper representations, not backing or another
            // arena. The existing admitted FailureBody remains counted once.
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(size_of::<Result<Self, GgufHostCopyCause>>()))
            .ok_or(GgufHostCopyCause::Layout)?;
        Ok(Self {
            controls,
            backing: plan.facts().backing_bytes(),
        })
    }
    fn total_bytes(self) -> Result<u64, GgufHostCopyCause> {
        self.controls
            .checked_add(self.backing)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(GgufHostCopyCause::Layout)
    }
}

// Pure native layout plus the existing native shape conversion, over the
// neutral converter's actual ordered outputs. No source-format equations here.
// Control-only diagnostics deliberately exclude the immutable backing.
pub(super) fn source_copy_control_bytes(
    plan: &GgufConversionPlan,
    runtime: &PreparedInputRuntime,
) -> Result<(u64, usize), GgufHostCopyCause> {
    source_copy_layouts(plan, runtime, None, false)
}
pub(super) fn source_copy_storage_bytes(
    plan: &GgufConversionPlan,
    runtime: &PreparedInputRuntime,
) -> Result<(u64, usize), GgufHostCopyCause> {
    source_copy_layouts(plan, runtime, None, true)
}
#[cfg(test)]
pub(super) fn source_copy_output_control_bytes(
    plan: &GgufConversionPlan,
    runtime: &PreparedInputRuntime,
    ordinal: usize,
) -> Result<u64, GgufHostCopyCause> {
    source_copy_output_bytes(plan, runtime, ordinal, false)
}
#[cfg(test)]
pub(super) fn source_copy_output_storage_bytes(
    plan: &GgufConversionPlan,
    runtime: &PreparedInputRuntime,
    ordinal: usize,
) -> Result<u64, GgufHostCopyCause> {
    source_copy_output_bytes(plan, runtime, ordinal, true)
}
#[cfg(test)]
fn source_copy_output_bytes(
    plan: &GgufConversionPlan,
    runtime: &PreparedInputRuntime,
    ordinal: usize,
    backing: bool,
) -> Result<u64, GgufHostCopyCause> {
    if ordinal >= plan.conversion().outputs().len() {
        return Err(GgufHostCopyCause::TypedBinding);
    }
    source_copy_layouts(plan, runtime, Some(ordinal), backing).map(|(bytes, _)| bytes)
}
fn source_copy_layouts(
    plan: &GgufConversionPlan,
    runtime: &PreparedInputRuntime,
    only: Option<usize>,
    backing: bool,
) -> Result<(u64, usize), GgufHostCopyCause> {
    fn one<T: safemlx::ArrayElement + Send + 'static>(
        runtime: &PreparedInputRuntime,
        shape: &[i32],
        capacity: usize,
    ) -> Result<SourceCopyStorage, GgufHostCopyCause> {
        let plan = OwnedHostCopyPlan::<T>::new(runtime, shape, capacity).map_err(|cause| {
            GgufHostCopyCause::Copy {
                cause,
                native: None,
            }
        })?;
        SourceCopyStorage::for_plan(&plan, true)
    }
    let mut sum = 0u64;
    let mut shape_storage = size_of::<bool>();
    for (ordinal, output) in plan.conversion().outputs().iter().enumerate() {
        if only.is_some_and(|selected| selected != ordinal) {
            continue;
        }
        let request = TypedOutputRequest::for_output(plan.conversion(), ordinal)
            .ok_or(GgufHostCopyCause::TypedBinding)?;
        let shape = gguf::mlx_shape_i32(plan.output_name(), output.shape())?;
        shape_storage = shape_storage.max(
            Layout::array::<i32>(shape.capacity())
                .map_err(|_| GgufHostCopyCause::Layout)?
                .size()
                .checked_add(size_of::<Vec<i32>>())
                .and_then(|n| n.checked_add(size_of::<bool>()))
                .ok_or(GgufHostCopyCause::Layout)?,
        );
        let capacity = request.output_elements;
        let storage = match request.dtype {
            LogicalDtype::F32 => one::<f32>(runtime, &shape, capacity),
            LogicalDtype::F16 => one::<half::f16>(runtime, &shape, capacity),
            LogicalDtype::Bf16 => one::<half::bf16>(runtime, &shape, capacity),
            LogicalDtype::I8 => one::<i8>(runtime, &shape, capacity),
            LogicalDtype::I16 => one::<i16>(runtime, &shape, capacity),
            LogicalDtype::I32 => one::<i32>(runtime, &shape, capacity),
            LogicalDtype::I64 => one::<i64>(runtime, &shape, capacity),
            LogicalDtype::F64 => one::<f64>(runtime, &shape, capacity),
            LogicalDtype::U8 => one::<u8>(runtime, &shape, capacity),
            LogicalDtype::U32 => one::<u32>(runtime, &shape, capacity),
        }?;
        let bytes = if backing {
            storage.total_bytes()?
        } else {
            u64::try_from(storage.controls).map_err(|_| GgufHostCopyCause::Layout)?
        };
        sum = sum.checked_add(bytes).ok_or(GgufHostCopyCause::Layout)?;
    }
    Ok((sum, shape_storage))
}

impl PendingWeightMaterialization {
    // Source banks are installed by the constructor and never extracted from
    // this owner. This tests construction mode, not the bank's remaining grant.
    pub(in crate::backend::runtime::checkpoint::store) fn has_gguf_cache_source_bank(
        &self,
    ) -> bool {
        self.retained
            .retention()
            .gguf_host
            .as_ref()
            .is_some_and(|host| host.source_constructions.is_some())
    }
    pub(in crate::backend::runtime::checkpoint::store) fn prepare_gguf_cache_key(
        &mut self,
    ) -> Result<(), PreparedGgufHostCopyFailure> {
        let value = self.retained.retention_mut();
        let host = value
            .gguf_host
            .as_mut()
            .expect("original GGUF pending owner");
        let WeightLeaseSource::Gguf(source) = &value.lease.source else {
            unreachable!("GGUF cache dispatch")
        };
        let result = match (&mut host.host_destinations, &mut host.source_constructions) {
            (Some(destinations), Some(bank)) => super::super::cache::PreparedEntry::prepare(
                source.lease.as_ref().identity(),
                destinations,
                bank,
            ),
            _ => Err((GgufHostCopyCause::TypedBinding, None)),
        };
        match result {
            Ok(entry) => {
                host.cache = Some(entry);
                Ok(())
            }
            Err((cause, entry)) => {
                host.cache = entry;
                // This refusal precedes G1. Move the same neutral Box into the
                // existing source holder before an error can outlive recovery.
                Err(self.bind_original_gguf_host().failure(cause))
            }
        }
    }
    pub(in crate::backend::runtime::checkpoint::store) fn complete_gguf_cache_entry(
        &mut self,
        arrays: CachedGgufArrays,
        index: &mut super::super::cache::Index,
    ) -> Result<
        (
            super::super::cache::Group,
            Option<super::super::cache::Node>,
        ),
        PreparedGgufHostCopyFailure,
    > {
        let host = self
            .retained
            .retention_mut()
            .gguf_host
            .as_mut()
            .expect("original GGUF pending owner");
        let entry = host
            .cache
            .as_mut()
            .expect("precharged current-miss metadata");
        entry.complete(arrays);
        let retired = match index.insert_or_replace(&mut entry.node, |row| row.stale()) {
            Ok(retired) => retired,
            Err(()) => return Err(host.failure(GgufHostCopyCause::TypedBinding)),
        };
        Ok((entry.take_group(), retired))
    }
}
