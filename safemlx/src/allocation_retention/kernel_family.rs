//! A finite immutable source family and its lazily populated platform slots.
use super::{OwnedNode, RetiredOwner, destroy, retire, take_owner};
use crate::{
    Array, Dtype, Stream,
    fast::{
        BorrowedKernelOutput, BorrowedKernelTemplate, MetalKernel, MetalKernelDefinitionPlan,
        fixed_config,
        prepared_definition::{KernelDefinitionCause, KernelDefinitionError, text},
    },
};
use std::{alloc::Layout, fmt, marker::PhantomData, mem::size_of, ptr};

/// The actual Metal source argument address-space class.
#[derive(Clone, Copy, Debug)]
pub enum KernelInputClass {
    /// Rank-zero input, passed as a constant reference.
    Scalar,
    /// Non-scalar input smaller than eight elements, passed as a constant pointer.
    Small,
    /// Other non-scalar input, passed as a device pointer.
    Device,
}
/// One declared input source signature, independent of its particular dimensions.
#[derive(Clone, Copy, Debug)]
pub struct KernelInputSignature {
    /// Actual scalar type.
    pub dtype: Dtype,
    /// Actual rank/element-count address-space class.
    pub class: KernelInputClass,
}
/// A closed source specialization. Names and values are copied at realization.
#[derive(Clone, Copy, Debug)]
pub struct KernelSpecialization<'a, const I: usize, const O: usize> {
    /// Inputs in the definition's declared order.
    pub inputs: [KernelInputSignature; I],
    /// Output scalar types in the definition's declared order.
    pub outputs: [Dtype; O],
    /// At most four fixed template values.
    pub templates: &'a [BorrowedKernelTemplate<'a>],
}
impl<const I: usize, const O: usize> KernelSpecialization<'_, I, O> {
    fn raw(&self) -> Result<safemlx_sys::mlx_fast_specialization, KernelDefinitionCause> {
        if I > 8 || O > 4 || self.templates.len() > 4 {
            return Err(KernelDefinitionCause::Invalid);
        }
        let mut raw = safemlx_sys::mlx_fast_specialization {
            inputs: [safemlx_sys::mlx_fast_input_signature {
                dtype: Dtype::Float32.into(),
                scalar: false,
                constant: false,
            }; 8],
            outputs: [Dtype::Float32.into(); 4],
            templates: [safemlx_sys::mlx_fast_template_view {
                name: ptr::null(),
                kind: 0,
                value: 0,
            }; 4],
            template_count: self.templates.len(),
        };
        for (target, input) in raw.inputs.iter_mut().zip(&self.inputs) {
            *target = safemlx_sys::mlx_fast_input_signature {
                dtype: input.dtype.into(),
                scalar: matches!(input.class, KernelInputClass::Scalar),
                constant: !matches!(input.class, KernelInputClass::Device),
            };
        }
        for (target, dtype) in raw.outputs.iter_mut().zip(&self.outputs) {
            *target = (*dtype).into();
        }
        for (target, value) in raw.templates.iter_mut().zip(self.templates) {
            *target = value.raw().ok_or(KernelDefinitionCause::Invalid)?;
        }
        Ok(raw)
    }
}
/// Exact native definition/family blocks, custody node and named controls.
#[derive(Clone, Copy, Debug)]
pub struct KernelFamilyLayout {
    definition: usize,
    family: usize,
    node: usize,
    controls: usize,
}
impl KernelFamilyLayout {
    /// All requested managed constructor storage. Platform allocator, shader
    /// compiler and driver internals are excluded; owned handles are included.
    pub fn required_bytes(self) -> Option<usize> {
        self.definition
            .checked_add(self.family)?
            .checked_add(self.node)?
            .checked_add(self.controls)
    }
    /// Immutable definition block request.
    pub fn definition_bytes(self) -> usize {
        self.definition
    }
    /// Finite source text, cache slots and their synchronization request.
    pub fn family_bytes(self) -> usize {
        self.family
    }
}
/// A finite family of distinct source signatures, compiled lazily into the
/// slots allocated at realization. This does not observe or promote old caches.
#[derive(Clone, Copy, Debug)]
pub struct MetalKernelFamilyPlan<'a, const I: usize, const O: usize, const S: usize> {
    /// Shared source definition.
    pub definition: MetalKernelDefinitionPlan<'a, I, O>,
    /// Exact declared source signatures; duplicate signatures are rejected.
    pub specializations: [KernelSpecialization<'a, I, O>; S],
}
impl<const I: usize, const O: usize, const S: usize> MetalKernelFamilyPlan<'_, I, O, S> {
    fn raw_specializations(
        &self,
    ) -> Result<[safemlx_sys::mlx_fast_specialization; S], KernelDefinitionCause> {
        if S == 0 || S > 64 {
            return Err(KernelDefinitionCause::Invalid);
        }
        // Fixed stack output; each entry is validated before the native call.
        let mut out = [self.specializations[0].raw()?; S];
        for (target, value) in out.iter_mut().zip(&self.specializations) {
            *target = value.raw()?;
        }
        Ok(out)
    }
    /// Pure exact requested layout. No source block, device or shader compiler
    /// is initialized by this query.
    pub fn layout<T: Send + 'static>(&self) -> Result<KernelFamilyLayout, KernelDefinitionCause> {
        let inputs = self.definition.inputs.map(text);
        let outputs = self.definition.outputs.map(text);
        let definition = self.definition.raw(&inputs, &outputs);
        let values = self.raw_specializations()?;
        let mut native = safemlx_sys::mlx_fast_kernel_family_layout::default();
        // SAFETY: all fixed output and borrowed descriptors remain live.
        let status = unsafe {
            safemlx_sys::mlx_fast_kernel_family_layout_for(
                &mut native,
                &definition,
                values.as_ptr(),
                S,
            )
        };
        if status != 0 {
            return Err(KernelDefinitionCause::status(status));
        }
        let controls = [
            size_of::<Self>(),
            size_of::<[safemlx_sys::mlx_fast_text_view; I]>(),
            size_of::<[safemlx_sys::mlx_fast_text_view; O]>(),
            size_of::<safemlx_sys::mlx_fast_definition_view>(),
            size_of::<[safemlx_sys::mlx_fast_specialization; S]>(),
            size_of::<Result<[safemlx_sys::mlx_fast_specialization; S], KernelDefinitionCause>>(),
            size_of::<safemlx_sys::mlx_fast_specialization>(),
            size_of::<Result<safemlx_sys::mlx_fast_specialization, KernelDefinitionCause>>(),
            size_of::<safemlx_sys::mlx_fast_kernel_family_layout>(),
            size_of::<KernelFamilyLayout>(),
            size_of::<Result<KernelFamilyLayout, KernelDefinitionCause>>(),
            size_of::<Option<usize>>(),
            size_of::<Layout>(),
            size_of::<*mut OwnedNode<T>>(),
            size_of::<Box<OwnedNode<T>>>(),
            size_of::<T>(),
            size_of::<super::RetirementBatch>(),
            size_of::<*mut RetiredOwner>(),
            size_of::<unsafe fn(*mut RetiredOwner)>(),
            size_of::<safemlx_sys::mlx_fast_kernel_family>(),
            size_of::<PreparedMetalKernelFamily<T>>(),
            size_of::<KernelDefinitionError<T>>(),
            size_of::<Result<PreparedMetalKernelFamily<T>, KernelDefinitionError<T>>>(),
            size_of::<i32>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(
                native
                    .control_bytes
                    .checked_add(std::mem::size_of_val(&controls))
                    .ok_or(KernelDefinitionCause::Overflow)?,
                usize::checked_add,
            )
            .ok_or(KernelDefinitionCause::Overflow)?;
        Ok(KernelFamilyLayout {
            definition: native.definition_bytes,
            family: native.family_bytes,
            node: size_of::<OwnedNode<T>>(),
            controls,
        })
    }
    /// Construct after admission. On failure every native prefix and node is
    /// freed before the unchanged owner is returned. No shader compilation runs.
    pub fn realize<T: Send + 'static>(
        self,
        owner: T,
    ) -> Result<PreparedMetalKernelFamily<T>, KernelDefinitionError<T>> {
        if let Err(cause) = self.layout::<T>() {
            return Err(KernelDefinitionError { cause, owner });
        }
        let values = match self.raw_specializations() {
            Ok(values) => values,
            Err(cause) => return Err(KernelDefinitionError { cause, owner }),
        };
        let inputs = self.definition.inputs.map(text);
        let outputs = self.definition.outputs.map(text);
        let definition = self.definition.raw(&inputs, &outputs);
        // SAFETY: nonzero concrete node layout; failure does not consume owner.
        let storage =
            unsafe { std::alloc::alloc(Layout::new::<OwnedNode<T>>()) }.cast::<OwnedNode<T>>();
        if storage.is_null() {
            return Err(KernelDefinitionError {
                cause: KernelDefinitionCause::AllocationFailed,
                owner,
            });
        }
        // SAFETY: exclusive aligned allocation, initialized exactly once.
        unsafe {
            storage.write(OwnedNode {
                retired: RetiredOwner {
                    next: ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
        }
        // SAFETY: initialized allocation belongs to this exact Box type.
        let node = unsafe { Box::from_raw(storage) };
        let mut raw = safemlx_sys::mlx_fast_kernel_family {
            ctx: ptr::null_mut(),
        };
        // SAFETY: native copies loans and consumes this queue node only on
        // success. Its eventual callback only enqueues, never executes T::drop.
        let status = unsafe {
            safemlx_sys::mlx_fast_kernel_family_new(
                &mut raw,
                &definition,
                values.as_ptr(),
                S,
                storage.cast(),
                Some(retire),
            )
        };
        if status != 0 {
            return Err(KernelDefinitionError {
                cause: KernelDefinitionCause::status(status),
                owner: take_owner(node),
            });
        }
        let _ = Box::into_raw(node);
        Ok(PreparedMetalKernelFamily {
            raw,
            owner: PhantomData,
        })
    }
}
/// Immutable generated source and finite lazy platform cache. Native primitives
/// retain this same family; the final native alias frees blocks and handles
/// before queuing the supplied owner for unlocked host reclamation.
pub struct PreparedMetalKernelFamily<T: Send + 'static> {
    raw: safemlx_sys::mlx_fast_kernel_family,
    owner: PhantomData<T>,
}
impl<T: Send + 'static> fmt::Debug for PreparedMetalKernelFamily<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedMetalKernelFamily")
            .finish_non_exhaustive()
    }
}
// SAFETY: source is immutable, cache access is native-mutex protected, aliases
// use native atomics. T is only moved to the unlocked retirement queue.
unsafe impl<T: Send + 'static> Send for PreparedMetalKernelFamily<T> {}
unsafe impl<T: Send + Sync + 'static> Sync for PreparedMetalKernelFamily<T> {}
impl<T: Send + 'static> Drop for PreparedMetalKernelFamily<T> {
    fn drop(&mut self) {
        // SAFETY: this is one owning native reference; no raw handle escapes.
        unsafe { safemlx_sys::mlx_fast_kernel_family_free(self.raw) };
    }
}
impl<T: Send + 'static> PreparedMetalKernelFamily<T> {
    /// Module-owned source preamble/catalog storage, charged once per domain.
    pub fn static_storage_bytes() -> usize {
        // SAFETY: pure static literal/type query.
        unsafe { safemlx_sys::mlx_fast_kernel_family_static_bytes() }
    }
    /// Actual fixed invocation and lazy-compilation control transports. Source
    /// text/cache slots are already funded by the family constructor.
    pub fn control_bytes<const I: usize, const O: usize>(
        templates: usize,
        rank: usize,
    ) -> Option<usize> {
        // SAFETY: pure native owner-layout query.
        let native = unsafe { safemlx_sys::mlx_fast_kernel_family_control_bytes() };
        if native == 0 {
            return None;
        }
        MetalKernel::fixed_control_bytes::<I, O>(templates, rank)?
            .checked_add(native)?
            .checked_add(size_of::<&Self>())
    }
    /// Invoke the shared worker, selecting only a declared signature. Borrowed
    /// invocation data never becomes cache-owned; outputs retain the family.
    pub fn apply_fixed_device<const I: usize, const O: usize>(
        &self,
        inputs: [&Array; I],
        outputs: [BorrowedKernelOutput<'_>; O],
        templates: &[BorrowedKernelTemplate<'_>],
        grid: [i32; 3],
        thread_group: [i32; 3],
        stream: &Stream,
    ) -> crate::error::Result<[Array; O]> {
        fixed_config::apply_fixed(
            fixed_config::FixedKernel::Family(self.raw),
            inputs,
            outputs,
            templates,
            grid,
            thread_group,
            stream,
        )
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
