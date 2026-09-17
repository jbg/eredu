//! Exact requested allocation layouts for the actual cloned declaration fields.
//! This describes cloning source rows, not map nodes, source/cache acquisitions,
//! native Array wrappers, arbitrary errors or allocator implementation overhead.
use super::*;
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::TensorSelection,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DeclarationCloneShape {
    pub(super) payload_bytes: usize,
    pub(super) allocations: usize,
}
impl DeclarationCloneShape {
    fn array<T>(&mut self, len: usize) -> Option<()> {
        let bytes = Layout::array::<T>(len).ok()?.size();
        self.payload_bytes = self.payload_bytes.checked_add(bytes)?;
        if bytes != 0 {
            self.allocations = self.allocations.checked_add(1)?;
        }
        Some(())
    }
    fn text(&mut self, value: &str) -> Option<()> {
        self.array::<u8>(value.len())
    }
    fn selection(&mut self, value: &TensorSelection) -> Option<()> {
        match value {
            TensorSelection::Full | TensorSelection::Range { .. } => Some(()),
            TensorSelection::Indices { indices, .. } => self.array::<usize>(indices.len()),
            TensorSelection::Contiguous { shape, .. } => self.array::<usize>(shape.len()),
        }
    }
    fn dtype(&mut self, value: &RecipeDtype) -> Option<()> {
        use RecipeDtype::*;
        match value {
            Other(value) => self.text(value),
            Bool | U8 | I8 | I16 | U16 | F16 | BF16 | I32 | U32 | F32 | F64 | I64 | U64 | C64
            | F8E4M3 | F8E5M2 | F4 | F8E8M0 => Some(()),
            _ => None, // Future non-exhaustive variants require their real shape.
        }
    }
    fn child(&mut self, child: &DerivedWeightRecipe) -> Option<()> {
        self.array::<DerivedWeightRecipe>(1)?; // Actual unary Box allocation.
        self.recipe(child)
    }
    fn recipe(&mut self, value: &DerivedWeightRecipe) -> Option<()> {
        use DerivedWeightRecipe::*;
        match value {
            Source { key, selection } => {
                self.text(key)?;
                self.selection(selection)
            }
            Select { input, selection } => {
                self.child(input)?;
                self.selection(selection)
            }
            Concatenate { inputs, .. } | Stack { inputs, .. } => {
                self.array::<DerivedWeightRecipe>(inputs.len())?;
                for input in inputs {
                    self.recipe(input)?;
                }
                Some(())
            }
            Reshape { input, shape } => {
                self.child(input)?;
                self.array::<usize>(shape.len())
            }
            Transpose { input, axes } => {
                self.child(input)?;
                self.array::<usize>(axes.len())
            }
            Cast { input, dtype } => {
                self.child(input)?;
                self.dtype(dtype)
            }
            View {
                input,
                dtype,
                shape,
            } => {
                self.child(input)?;
                self.dtype(dtype)?;
                self.array::<usize>(shape.len())
            }
            NegLog { input } | SubtractOne { input } => self.child(input),
        }
    }
    fn binding(&mut self, binding: &WeightBinding) -> Option<()> {
        self.text(binding.name())?;
        self.text(binding.checkpoint_key())?;
        if let Some(alias) = binding.alias_of() {
            self.text(alias)?;
        }
        if let Some(target) = binding.logical_target() {
            self.text(target)?;
        }
        self.selection(binding.selection())?;
        if let Some(recipe) = binding.recipe() {
            self.recipe(recipe)?;
        }
        if let Some(companions) = binding.quantization_companions() {
            self.text(companions.scale())?;
            if let Some(bias) = companions.affine_bias() {
                self.text(bias)?;
            }
        }
        Some(())
    }
    pub(super) fn binding_payload(binding: &WeightBinding) -> Option<usize> {
        let mut value = Self::default();
        value.binding(binding)?;
        Some(value.payload_bytes)
    }
    pub(super) fn bindings(bindings: &[WeightBinding]) -> Option<Self> {
        let mut value = Self::default();
        value.array::<WeightBinding>(bindings.len())?;
        for binding in bindings {
            value.binding(binding)?;
        }
        Some(value)
    }
}
