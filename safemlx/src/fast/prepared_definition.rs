//! One immutable native definition block, retired before its supplied owner.
use super::{fixed_config, BorrowedKernelOutput, BorrowedKernelTemplate};
use crate::{Array, Stream};
use std::{fmt, mem::size_of};

/// Fixed constructor failure. No native formatted error or heap error owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelDefinitionCause {
    /// The actual native layout is not qualified on this toolchain.
    UnknownLayout,
    /// The fixed input/output population is outside the supported constructor.
    Invalid,
    /// Exact requested storage cannot be represented.
    Overflow,
    /// The one native requested allocation failed.
    AllocationFailed,
}
impl fmt::Display for KernelDefinitionCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnknownLayout => "Metal kernel definition layout is unqualified",
            Self::Invalid => "invalid fixed Metal kernel definition",
            Self::Overflow => "Metal kernel definition storage overflow",
            Self::AllocationFailed => "Metal kernel definition allocation failed",
        })
    }
}
impl std::error::Error for KernelDefinitionCause {}
impl KernelDefinitionCause {
    pub(crate) fn status(status: i32) -> Self {
        match status {
            1 => Self::UnknownLayout,
            3 => Self::Overflow,
            4 => Self::AllocationFailed,
            _ => Self::Invalid,
        }
    }
}

/// Owning refusal; the supplied constructor custody remains intact.
#[derive(Debug)]
pub struct KernelDefinitionError<T> {
    pub(crate) cause: KernelDefinitionCause,
    pub(crate) owner: T,
}
impl<T> KernelDefinitionError<T> {
    /// Fixed native or layout refusal.
    pub fn cause(&self) -> KernelDefinitionCause {
        self.cause
    }
    /// Return the unchanged owner after a refused constructor.
    pub fn into_parts(self) -> (KernelDefinitionCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Display for KernelDefinitionError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T: fmt::Debug> std::error::Error for KernelDefinitionError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Actual one-block request and simultaneous constructor/control transports.
#[derive(Clone, Copy, Debug)]
pub struct KernelDefinitionLayout {
    bytes: usize,
    alignment: usize,
    controls: usize,
}
impl KernelDefinitionLayout {
    /// Requested native block bytes, including its fixed header and copied text.
    pub fn requested_bytes(self) -> usize {
        self.bytes
    }
    /// Native allocation alignment (the header has ordinary new alignment).
    pub fn alignment(self) -> usize {
        self.alignment
    }
    /// The block plus its construction/control contribution; no generated
    /// source, primitive, physical output or library/cache owner is included.
    pub fn required_bytes(self) -> Option<usize> {
        self.bytes.checked_add(self.controls)
    }
}

/// Borrowed cold constructor description. Realization copies all text into
/// the immutable native block; no Rust/C string or Vec allocation is needed.
#[derive(Clone, Copy, Debug)]
pub struct MetalKernelDefinitionPlan<'a, const I: usize, const O: usize> {
    /// Native function name.
    pub name: &'a str,
    /// Actual source body.
    pub source: &'a str,
    /// Shared source preamble.
    pub header: &'a str,
    /// Actual input argument names, in invocation order.
    pub inputs: [&'a str; I],
    /// Actual output argument names, in invocation order.
    pub outputs: [&'a str; O],
    /// Preserve the ordinary kernel's contiguous-input policy.
    pub ensure_row_contiguous: bool,
    /// Preserve the ordinary kernel's atomic-output policy.
    pub atomic_outputs: bool,
}
pub(crate) fn text(value: &str) -> safemlx_sys::mlx_fast_text_view {
    safemlx_sys::mlx_fast_text_view {
        data: value.as_ptr().cast(),
        size: value.len(),
    }
}
impl<const I: usize, const O: usize> MetalKernelDefinitionPlan<'_, I, O> {
    pub(crate) fn raw(
        &self,
        inputs: &[safemlx_sys::mlx_fast_text_view; I],
        outputs: &[safemlx_sys::mlx_fast_text_view; O],
    ) -> safemlx_sys::mlx_fast_definition_view {
        safemlx_sys::mlx_fast_definition_view {
            name: text(self.name),
            source: text(self.source),
            header: text(self.header),
            inputs: inputs.as_ptr(),
            input_count: I,
            outputs: outputs.as_ptr(),
            output_count: O,
            ensure_row_contiguous: self.ensure_row_contiguous,
            atomic_outputs: self.atomic_outputs,
        }
    }
    /// Pure qualified request before custody admission or allocation.
    pub fn layout<T>(&self) -> Result<KernelDefinitionLayout, KernelDefinitionCause> {
        let inputs = self.inputs.map(text);
        let outputs = self.outputs.map(text);
        let raw = self.raw(&inputs, &outputs);
        let mut native = safemlx_sys::mlx_fast_definition_layout::default();
        // SAFETY: all byte/name loans remain live; native only checks and sums.
        let status = unsafe { safemlx_sys::mlx_fast_metal_definition_layout(&mut native, &raw) };
        if status != 0 {
            return Err(KernelDefinitionCause::status(status));
        }
        let controls = [
            size_of::<Self>(),
            size_of::<[safemlx_sys::mlx_fast_text_view; I]>(),
            size_of::<[safemlx_sys::mlx_fast_text_view; O]>(),
            size_of::<safemlx_sys::mlx_fast_definition_view>(),
            size_of::<safemlx_sys::mlx_fast_definition_layout>(),
            size_of::<KernelDefinitionLayout>(),
            size_of::<Result<KernelDefinitionLayout, KernelDefinitionCause>>(),
            size_of::<Option<usize>>(),
            size_of::<safemlx_sys::mlx_fast_prepared_definition>(),
            size_of::<PreparedMetalKernelDefinition<T>>(),
            size_of::<KernelDefinitionError<T>>(),
            size_of::<Result<PreparedMetalKernelDefinition<T>, KernelDefinitionError<T>>>(),
            size_of::<i32>(),
        ]
        .into_iter()
        .try_fold(native.construction_controls, usize::checked_add)
        .ok_or(KernelDefinitionCause::Overflow)?;
        Ok(KernelDefinitionLayout {
            bytes: native.requested_bytes,
            alignment: native.alignment,
            controls,
        })
    }
    /// Realize with an already admitted constructor owner. Failure returns it
    /// intact. Successful definition storage is always freed before that owner.
    pub fn realize<T>(
        self,
        owner: T,
    ) -> Result<PreparedMetalKernelDefinition<T>, KernelDefinitionError<T>> {
        if let Err(cause) = self.layout::<T>() {
            return Err(KernelDefinitionError { cause, owner });
        }
        let inputs = self.inputs.map(text);
        let outputs = self.outputs.map(text);
        let raw = self.raw(&inputs, &outputs);
        let mut definition = safemlx_sys::mlx_fast_prepared_definition {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: native validates and copies every loan synchronously. It
        // returns either one unique immutable block or an empty output handle.
        let status = unsafe { safemlx_sys::mlx_fast_metal_definition_new(&mut definition, &raw) };
        if status != 0 {
            return Err(KernelDefinitionError {
                cause: KernelDefinitionCause::status(status),
                owner,
            });
        }
        Ok(PreparedMetalKernelDefinition {
            raw: definition,
            _owner: owner,
        })
    }
}

/// Immutable native definition plus its original constructor custody. No raw
/// handle, Weak, or clone grant escapes. Outputs own generated text separately.
pub struct PreparedMetalKernelDefinition<T> {
    raw: safemlx_sys::mlx_fast_prepared_definition,
    _owner: T,
}
impl<T> fmt::Debug for PreparedMetalKernelDefinition<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedMetalKernelDefinition")
            .finish_non_exhaustive()
    }
}
// SAFETY: the block is immutable after construction; its pointers refer only
// into that same allocation. Apply borrows it synchronously and generates
// independent output owners. T's corresponding thread property is preserved.
unsafe impl<T: Send> Send for PreparedMetalKernelDefinition<T> {}
unsafe impl<T: Sync> Sync for PreparedMetalKernelDefinition<T> {}
impl<T> Drop for PreparedMetalKernelDefinition<T> {
    fn drop(&mut self) {
        // SAFETY: this unique native block has no external raw/weak owners.
        // Its destruction performs no callbacks or native runtime locking.
        unsafe { safemlx_sys::mlx_fast_metal_definition_free(self.raw) };
        // Rust then drops _owner, after the block and its inline text retire.
    }
}
impl<T> PreparedMetalKernelDefinition<T> {
    /// Actual native attribute catalog storage. This is fixed module storage,
    /// to be included once in the domain baseline, never once per invocation.
    pub fn static_storage_bytes() -> usize {
        // SAFETY: a pure sizeof query; no initialization or allocation.
        unsafe { safemlx_sys::mlx_fast_metal_definition_static_bytes() }
    }
    /// Apply through the same fixed invocation and ordinary native generator.
    /// Generated-source and cache custody are separate prerequisites.
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
            fixed_config::FixedKernel::Prepared(self.raw),
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
mod tests {
    use super::super::{CustomKernelConfig, MetalKernel};
    use super::*;
    use crate::{Device, DeviceType, Dtype};
    #[test]
    fn definition_copies_text_and_outputs_survive_definition_retirement() {
        let mut body = String::from("uint i=thread_position_in_grid.x; output[i]=input[i]*GAIN;");
        let plan = MetalKernelDefinitionPlan {
            name: "copied_definition",
            source: &body,
            header: "",
            inputs: ["input"],
            outputs: ["output"],
            ensure_row_contiguous: true,
            atomic_outputs: false,
        };
        let qualified = plan.layout::<()>().is_ok();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_KERNEL_DEFINITION").is_some() {
            assert!(qualified, "pinned kernel definition must qualify");
        }
        if !qualified {
            return;
        }
        let definition = plan.realize(()).unwrap();
        let ordinary = MetalKernel::new(
            plan.name,
            plan.inputs,
            plan.outputs,
            plan.source,
            plan.header,
            true,
            false,
        )
        .unwrap();
        // The native definition must have copied this source, not retained its loan.
        body.clear();
        body.push_str("this is not Metal code");
        drop(body);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let input = Array::try_from_slice(&[-2.0f32, -0.25, 0.5, 3.0, 7.0, 11.0, 13.0, 17.0], &[8])
            .unwrap();
        let outputs = [BorrowedKernelOutput {
            shape: &[8],
            dtype: Dtype::Float32,
        }];
        let [first] = definition
            .apply_fixed_device(
                [&input],
                outputs,
                &[BorrowedKernelTemplate::Int(c"GAIN", 3)],
                [8, 1, 1],
                [8, 1, 1],
                &stream,
            )
            .unwrap();
        let [second] = definition
            .apply_fixed_device(
                [&input],
                outputs,
                &[BorrowedKernelTemplate::Int(c"GAIN", -2)],
                [8, 1, 1],
                [8, 1, 1],
                &stream,
            )
            .unwrap();
        // Both deferred primitives own their generated source; evaluation after
        // the constructor block retires must not read its old pointers.
        drop(definition);
        for (gain, actual) in [(3, first), (-2, second)] {
            let config = CustomKernelConfig::new()
                .with_template_arg_int("GAIN", gain)
                .with_grid([8, 1, 1])
                .with_thread_group([8, 1, 1])
                .with_output_arg([8], Dtype::Float32);
            let expected = ordinary
                .apply_one_device([&input], &config, &stream)
                .unwrap();
            assert_eq!(
                crate::array::eval_vec::<f32>(&actual),
                crate::array::eval_vec::<f32>(&expected)
            );
            let values = crate::array::eval_vec::<f32>(&actual);
            assert_eq!(values[0], -2.0 * gain as f32);
            assert_eq!(values[7], 17.0 * gain as f32);
        }
        let owner = std::sync::Arc::new(());
        let rejected = MetalKernelDefinitionPlan {
            name: "invalid",
            source: "",
            header: "",
            inputs: [],
            outputs: [],
            ensure_row_contiguous: true,
            atomic_outputs: false,
        }
        .realize(owner.clone())
        .unwrap_err();
        assert_eq!(std::sync::Arc::strong_count(&owner), 2);
        let (cause, returned) = rejected.into_parts();
        assert_eq!(cause, KernelDefinitionCause::Invalid);
        assert!(std::sync::Arc::ptr_eq(&owner, &returned));
        drop(returned);
        assert_eq!(std::sync::Arc::strong_count(&owner), 1);
    }
}
