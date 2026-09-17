//! Cold CPU matrix implementation choice, independent of native contexts.

/// Numerical implementation requested before constructing a CPU equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuMatmulImplementation {
    /// Preserve the backend's ordinary platform choice.
    PlatformDefault,
    /// F32 products accumulated in fixed square tiles and SIMD partial sums.
    Float32Tiles,
    /// F32 or F16 inputs using the same fixed tiles and F32 SIMD accumulation.
    Float32AndFloat16Tiles,
}
/// Descriptive facts of compiled fixed tiles with F32 accumulation.
/// This carries no source, allocation or execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuMatmulFacts {
    tile_edge: u32,
    reduction_lanes: u32,
    max_rank: u32,
    max_elements: u32,
    float16_tiles: bool,
    platform_float16_tiles: bool,
}
impl CpuMatmulFacts {
    /// Validate the backend's finite geometry and reduction declaration.
    pub const fn new(tile_edge: u32, reduction_lanes: u32, max_rank: u32, max_elements: u32) -> Option<Self> {
        if tile_edge==0 || reduction_lanes==0 || reduction_lanes>tile_edge ||
            tile_edge%reduction_lanes!=0 || max_rank<2 || max_elements<tile_edge {
            return None;
        }
        Some(Self {tile_edge,reduction_lanes,max_rank,max_elements,float16_tiles:false,platform_float16_tiles:false})
    }
    /// Attach compiled F16 input support and the actual default F16 branch.
    /// A platform SIMD claim requires that same worker to be available.
    pub const fn with_float16_tiles(mut self, supported:bool, platform_default:bool)->Option<Self> {
        if platform_default&&!supported { return None; }
        self.float16_tiles=supported;self.platform_float16_tiles=platform_default;Some(self)
    }
    /// The compiled tile worker accepts F16 inputs with F32 accumulation.
    pub const fn float16_tiles(self)->bool {self.float16_tiles}
    /// The ordinary platform F16 branch is exactly that tile worker.
    pub const fn platform_float16_tiles(self)->bool {self.platform_float16_tiles}
    /// Accumulation tile edge; final incomplete tiles use the same worker.
    pub const fn tile_edge(self) -> u32 { self.tile_edge }
    /// Values reduced together before adding the partial sum to the tile.
    pub const fn reduction_lanes(self) -> u32 { self.reduction_lanes }
    /// Maximum normalized rank of both broadcast matrix operands.
    pub const fn max_rank(self) -> u32 { self.max_rank }
    /// Signed worker index ceiling for each whole broadcast operand/output.
    pub const fn max_elements(self) -> u32 { self.max_elements }
    /// Retain this exact source declaration as the selected implementation.
    pub const fn select(self) -> SelectedCpuMatmul {
        SelectedCpuMatmul {implementation:CpuMatmulImplementation::Float32Tiles,facts:Some(self)}
    }
    /// Select explicit F32/F16 tiles only from a compiled F16 source fact.
    pub const fn select_with_float16(self)->Option<SelectedCpuMatmul> {
        if !self.float16_tiles {return None;}
        Some(SelectedCpuMatmul {implementation:CpuMatmulImplementation::Float32AndFloat16Tiles,facts:Some(self)})
    }
    /// Retain observed platform facts without asserting an F32 BLAS bound.
    pub const fn select_platform_default(self)->SelectedCpuMatmul {
        SelectedCpuMatmul {implementation:CpuMatmulImplementation::PlatformDefault,facts:Some(self)}
    }
}
/// Immutable cold choice; the backend must still authenticate its native source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedCpuMatmul {
    implementation: CpuMatmulImplementation,
    facts: Option<CpuMatmulFacts>,
}
impl SelectedCpuMatmul {
    /// Ordinary platform behavior has no inferred fixed-tile storage bound.
    pub const fn platform_default() -> Self {
        Self {implementation:CpuMatmulImplementation::PlatformDefault,facts:None}
    }
    /// Numerical implementation retained by this choice.
    pub const fn implementation(self) -> CpuMatmulImplementation { self.implementation }
    /// Exact finite source facts, absent for unspecified platform behavior.
    pub const fn facts(self) -> Option<CpuMatmulFacts> { self.facts }
    /// Validate normalized broadcast geometry without allocating or visiting data.
    /// Physical layout and input custody remain separate backend obligations.
    pub fn geometry(self, rank:u32, m:u32, n:u32, k:u32, batches:u32) -> Result<CpuMatmulGeometry,CpuMatmulError> {
        if self.implementation==CpuMatmulImplementation::PlatformDefault {return Err(CpuMatmulError::Unspecified);}
        self.geometry_from_facts(rank,m,n,k,batches)
    }
    /// F16 geometry is qualified either by explicit selection or by the exact
    /// compiled default SIMD branch. No platform F32 behavior is inferred.
    pub fn float16_geometry(self,rank:u32,m:u32,n:u32,k:u32,batches:u32)->Result<CpuMatmulGeometry,CpuMatmulError> {
        let f=self.facts.ok_or(CpuMatmulError::Unspecified)?;
        if !f.float16_tiles || !(self.implementation==CpuMatmulImplementation::Float32AndFloat16Tiles||f.platform_float16_tiles) {
            return Err(CpuMatmulError::Unspecified);
        }
        self.geometry_from_facts(rank,m,n,k,batches)
    }
    fn geometry_from_facts(self,rank:u32,m:u32,n:u32,k:u32,batches:u32)->Result<CpuMatmulGeometry,CpuMatmulError> {
        let f=self.facts.ok_or(CpuMatmulError::Unspecified)?;
        if rank<2||rank>f.max_rank||m==0||n==0||k==0||batches==0||(rank==2&&batches!=1) {
            return Err(CpuMatmulError::Geometry);
        }
        // Existing ceildiv adds tile_edge-1 before division.
        let ceiling=f.max_elements-(f.tile_edge-1);
        if m>ceiling||n>ceiling {return Err(CpuMatmulError::Overflow);}
        let product=|a:u32,b:u32| a.checked_mul(b).and_then(|x|x.checked_mul(batches))
            .filter(|&x|x<=f.max_elements).ok_or(CpuMatmulError::Overflow);
        Ok(CpuMatmulGeometry {rank,m,n,k,batches,left:product(m,k)?,right:product(k,n)?,output:product(m,n)?})
    }
}
/// Validated logical geometry of the selected fixed-tile worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuMatmulGeometry { rank:u32,m:u32,n:u32,k:u32,batches:u32,left:u32,right:u32,output:u32 }
impl CpuMatmulGeometry {
    /// Normalized rank, rows, columns, reduction width, and broadcast batches.
    pub const fn dimensions(self)->[u32;5] { [self.rank,self.m,self.n,self.k,self.batches] }
    /// Complete logical left, right, and output element populations.
    pub const fn elements(self)->[u32;3] { [self.left,self.right,self.output] }
}
/// Fixed cold selection/geometry refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CpuMatmulError {
    /// The platform choice did not declare a source-visible fixed-tile worker.
    #[error("CPU matrix implementation has no fixed-tile facts")]
    Unspecified,
    /// Empty or incompatible normalized matrix geometry.
    #[error("invalid selected CPU matrix geometry")]
    Geometry,
    /// The selected worker's signed index or tile range would overflow.
    #[error("selected CPU matrix geometry exceeds its index domain")]
    Overflow,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cold_cpu_matmul_choice_retains_reduction_and_checked_geometry() {
        let facts=CpuMatmulFacts::new(16,8,4,i32::MAX as u32).unwrap();
        let choice=facts.select();
        assert_eq!(choice.facts(),Some(facts));
        assert_eq!(choice.implementation(),CpuMatmulImplementation::Float32Tiles);
        assert_eq!(choice.geometry(3,17,19,23,2).unwrap().elements(),[782,874,646]);
        assert_eq!(choice.geometry(3,17,19,23,2).unwrap().dimensions(),[3,17,19,23,2]);
        assert_eq!(SelectedCpuMatmul::platform_default().geometry(2,2,3,4,1),Err(CpuMatmulError::Unspecified));
        assert_eq!(choice.geometry(2,2,3,4,2),Err(CpuMatmulError::Geometry));
        assert_eq!(choice.geometry(3,1,1,1,0),Err(CpuMatmulError::Geometry));
        assert_eq!(choice.geometry(5,2,3,4,1),Err(CpuMatmulError::Geometry));
        assert_eq!(choice.geometry(2,i32::MAX as u32,1,1,1),Err(CpuMatmulError::Overflow));
        assert_eq!(choice.geometry(3,65536,65536,1,1),Err(CpuMatmulError::Overflow));
        assert!(CpuMatmulFacts::new(16,3,4,i32::MAX as u32).is_none());
        assert!(CpuMatmulFacts::new(0,0,4,i32::MAX as u32).is_none());
    }
    #[test]
    fn half_choice_distinguishes_explicit_tiles_from_actual_platform_branch() {
        let base=CpuMatmulFacts::new(16,4,5,i32::MAX as u32).unwrap();
        assert!(base.select_with_float16().is_none());
        assert!(base.with_float16_tiles(false,true).is_none());
        for platform in [false,true] {
            let facts=base.with_float16_tiles(true,platform).unwrap();
            let selected=facts.select_with_float16().unwrap();
            assert_eq!(selected.implementation(),CpuMatmulImplementation::Float32AndFloat16Tiles);
            assert_eq!(selected.float16_geometry(5,19,17,23,12).unwrap().elements(),[5244,4692,3876]);
            assert_eq!(facts.select().float16_geometry(5,19,17,23,12).is_ok(),platform);
            assert_eq!(facts.select_platform_default().float16_geometry(5,19,17,23,12).is_ok(),platform);
            assert_eq!(facts.select_platform_default().geometry(2,2,3,4,1),Err(CpuMatmulError::Unspecified));
            assert_eq!(selected.float16_geometry(6,19,17,23,12),Err(CpuMatmulError::Geometry));
            assert_eq!(selected.float16_geometry(5,19,17,23,u32::MAX),Err(CpuMatmulError::Overflow));
        }
    }

}
