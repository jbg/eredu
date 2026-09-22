//! Pure CPU implementation selection and a separately paid native context plan.
use eredu_nn::{CpuMatmulFacts, CpuMatmulImplementation, SelectedCpuMatmul};
use safemlx::{CpuMatmulKernel, Stream, StreamCopyCause, StreamCopyPlan};
use std::mem::{size_of, size_of_val};

/// A cold selected CPU Matmul implementation; it owns no native resources and
/// grants no full-model, parameter, workspace or submission authority.
#[derive(Clone, Copy, Debug)]
pub struct MlxCpuMatmulMechanism {
    selected: SelectedCpuMatmul,
    native: Option<safemlx::CpuMatmulFacts>,
}
impl MlxCpuMatmulMechanism {
    /// Translate pure compiled worker facts into the neutral immutable choice.
    pub fn select(implementation: CpuMatmulImplementation) -> Option<Self> {
        let Some(native) = safemlx::CpuMatmulFacts::inspect() else {
            return (implementation == CpuMatmulImplementation::PlatformDefault).then_some(Self {
                selected: SelectedCpuMatmul::platform_default(),
                native: None,
            });
        };
        let facts = CpuMatmulFacts::new(
            u32::try_from(native.tile_edge()).ok()?,
            u32::try_from(native.reduction_lanes()).ok()?,
            u32::try_from(native.max_rank()).ok()?,
            u32::try_from(native.max_elements()).ok()?,
        )?
        .with_float16_tiles(native.float16_tiles(), native.platform_float16_tiles())?;
        let selected = match implementation {
            CpuMatmulImplementation::PlatformDefault => facts.select_platform_default(),
            CpuMatmulImplementation::Float32Tiles => facts.select(),
            CpuMatmulImplementation::Float32AndFloat16Tiles => facts.select_with_float16()?,
        };
        Some(Self {
            selected,
            native: Some(native),
        })
    }
    /// The exact neutral implementation and its reduction/geometry declaration.
    pub const fn selected(self) -> SelectedCpuMatmul {
        self.selected
    }
    /// Bind only this choice to a copied scalar CPU context. Realization uses
    /// the existing caller-funded StreamCopyPlan owner; the source is unchanged.
    pub fn stream_plan<T: Send + Sync + 'static>(
        self,
        source: &Stream,
    ) -> Result<StreamCopyPlan<T>, StreamCopyCause> {
        let kernel = match self.selected.implementation() {
            CpuMatmulImplementation::PlatformDefault => CpuMatmulKernel::PlatformDefault,
            CpuMatmulImplementation::Float32Tiles => CpuMatmulKernel::Float32Tiles,
            CpuMatmulImplementation::Float32AndFloat16Tiles => {
                CpuMatmulKernel::Float32AndFloat16Tiles
            }
        };
        StreamCopyPlan::capture(source)?.with_cpu_matmul(kernel)
    }
    /// Existing fixed Eval worker storage for this selected logical geometry.
    /// Native source validation additionally requires actual compact/transposed
    /// interiors; this partial fact does not admit a complete CPU equation.
    pub fn eval_layout(
        self,
        geometry: eredu_nn::CpuMatmulGeometry,
        tracer: bool,
    ) -> Option<safemlx::CpuCopyEvalLayout> {
        let [rank, m, n, k, batches] = geometry.dimensions();
        if self.selected.geometry(rank, m, n, k, batches).ok() != Some(geometry) {
            return None;
        }
        safemlx::OperationEvent::cpu_tiled_matmul_layout(
            rank as usize,
            m as usize,
            n as usize,
            k as usize,
            batches as usize,
            tracer,
        )
    }
    /// Existing selected worker plus at most two real matrix-interior copies.
    /// The exact primitive still validates readable source strides and counts
    /// which operands require the shared fallback at execution.
    pub fn eval_layout_with_copies(
        self,
        geometry: eredu_nn::CpuMatmulGeometry,
        copies: usize,
        tracer: bool,
    ) -> Option<safemlx::CpuCopyEvalLayout> {
        let [rank, m, n, k, batches] = geometry.dimensions();
        if self.selected.geometry(rank, m, n, k, batches).ok() != Some(geometry) {
            return None;
        }
        safemlx::OperationEvent::cpu_tiled_matmul_copy_layout(
            rank as usize,
            m as usize,
            n as usize,
            k as usize,
            batches as usize,
            copies,
            tracer,
        )
    }
    /// Existing F16 SIMD worker under the exact retained choice/default fact.
    pub fn float16_eval_layout(
        self,
        geometry: eredu_nn::CpuMatmulGeometry,
        copies: usize,
        tracer: bool,
    ) -> Option<safemlx::CpuCopyEvalLayout> {
        let [rank, m, n, k, batches] = geometry.dimensions();
        if self.selected.float16_geometry(rank, m, n, k, batches).ok() != Some(geometry) {
            return None;
        }
        safemlx::OperationEvent::cpu_f16_matmul_copy_layout(
            rank as usize,
            m as usize,
            n as usize,
            k as usize,
            batches as usize,
            copies,
            tracer,
        )
    }
    /// Concrete cold selection and context-plan controls, separate from the
    /// plan's own native/shared wrapper and retained source storage.
    pub fn control_bytes<T: Send + Sync + 'static>(self) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<SelectedCpuMatmul>(),
            size_of::<Option<CpuMatmulFacts>>(),
            size_of::<CpuMatmulImplementation>(),
            size_of::<CpuMatmulKernel>(),
            size_of::<StreamCopyPlan<T>>(),
            size_of::<Result<StreamCopyPlan<T>, StreamCopyCause>>(),
            size_of::<&Stream>(),
            size_of::<Result<u32, std::num::TryFromIntError>>() * 4,
            size_of::<eredu_nn::CpuMatmulGeometry>(),
            size_of::<[u32; 5]>(),
            size_of::<Result<eredu_nn::CpuMatmulGeometry, eredu_nn::CpuMatmulError>>(),
            size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<bool>() * 3,
            size_of::<Option<CpuMatmulFacts>>(),
            size_of::<Option<SelectedCpuMatmul>>(),
        ];
        let native = match self.native {
            Some(f) => f.control_bytes()?,
            None => 0,
        };
        parts
            .into_iter()
            .try_fold(native.checked_add(size_of_val(&parts))?, usize::checked_add)
    }
}
