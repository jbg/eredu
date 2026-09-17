//! Closed metadata clone producers. No public type can assert a clone footprint.
//!
//! The participating types use their existing Clone worker after the context
//! debit. String/Vec clones request exactly their logical length; nested policy
//! fields are counted recursively. No provider callback or arbitrary Clone is
//! admitted by a public API. These objects carry semantic metadata, never a
//! tensor allocation, source identity witness, or grant.
use super::{Error, WorkspaceContext, WorkspaceMetadataError};
use crate::*;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

pub(crate) trait MetadataClone: Clone {
    fn metadata_clone_bytes(&self) -> Option<usize>;
}

pub(crate) fn controls<T>() -> Option<usize> {
    let parts = [
        size_of::<&T>(),
        size_of::<T>(),
        size_of::<Option<usize>>(),
        size_of::<Result<T, Error>>(),
        size_of::<Result<(), WorkspaceMetadataError>>(),
        size_of::<(&T, &WorkspaceContext, usize)>(),
        size_of::<Layout>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl WorkspaceContext {
    pub(crate) fn clone_metadata<T: MetadataClone>(&self, source: &T) -> Result<T, Error> {
        let bytes = source
            .metadata_clone_bytes()
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.charge_metadata(bytes)?;
        Ok(source.clone())
    }

    // This boxes an already-owned value. Its nested storage has a separate
    // producer debit, so no second backing charge is added here.
    pub(crate) fn box_metadata<T>(&self, value: T) -> Result<Box<T>, Error> {
        let bytes = Layout::new::<T>()
            .size()
            .checked_add(controls::<Box<T>>().ok_or(WorkspaceMetadataError::Overflow)?)
            .and_then(|bytes| bytes.checked_add(size_of::<T>()))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.charge_metadata(bytes)?;
        Ok(Box::new(value))
    }
}

impl MetadataClone for String {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        controls::<Self>()?.checked_add(Layout::array::<u8>(self.len()).ok()?.size())
    }
}
impl<T: MetadataClone> MetadataClone for Option<T> {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        controls::<Self>()?.checked_add(match self {
            Some(value) => value.metadata_clone_bytes()?,
            None => 0,
        })
    }
}
impl<T: MetadataClone> MetadataClone for Vec<T> {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        let initial =
            controls::<Self>()?.checked_add(Layout::array::<T>(self.len()).ok()?.size())?;
        self.iter().try_fold(initial, |bytes, value| {
            bytes.checked_add(value.metadata_clone_bytes()?)
        })
    }
}
impl MetadataClone for u32 {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        Some(0)
    }
}
impl MetadataClone for f32 {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        Some(0)
    }
}
impl MetadataClone for ParameterId {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        controls::<Self>()?.checked_add(self.0.metadata_clone_bytes()?)
    }
}

// Exhaustive field patterns keep this census complete for each record: a new
// field cannot silently inherit its existing footprint.
macro_rules! record {
    ($ty:ty, { $($field:ident),* $(,)? }, [ $($owned:ident),* $(,)? ]) => {
        impl MetadataClone for $ty {
            fn metadata_clone_bytes(&self) -> Option<usize> {
                let Self { $($field),* } = self;
                $(let _ = $field;)*
                let bytes = controls::<Self>()?;
                $(let bytes = bytes.checked_add($owned.metadata_clone_bytes()?)?;)*
                Some(bytes)
            }
        }
    };
}
record!(ParameterSpec, {
    id, trainable, alias_of, group, linear_companion, linear_companion_of, linear_row_layout
}, [id, alias_of, group, linear_companion_of]);
record!(LinearFormatSpec, { format, scale, affine_bias, row_layout }, [scale, affine_bias]);
record!(NormalizationConstructionSpec, { groups, dimensions, epsilon, scale }, [scale]);
record!(SelectorInputTransformSpec, { epsilon, scale, inverse_sqrt_dimensions }, [scale]);
record!(TopKGroupSelectorSpec, {
    input_dimensions, weight, bias, correction_bias, input_transform, coefficient_scale,
    format, selection, arithmetic
}, [weight, bias, correction_bias, input_transform, coefficient_scale, format]);
record!(GroupedProjectionSpec, { weight, bias, format }, [weight, bias, format]);
record!(GatedProductGroupParameters, { gate, up, down }, [gate, up, down]);
record!(GroupedGatedProductSpec, {
    group_count, input_dimensions, intermediate_dimensions, output_dimensions, policy, reduction, layout
}, [layout]);
record!(GroupedRelu2Spec, {
    group_count, hidden_dimensions, intermediate_dimensions, up, down
}, [up, down]);
record!(HyperConnectionSpec, {
    streams, hidden_size, sinkhorn_iterations, epsilon, function, base, scale
}, [function, base, scale]);
record!(HyperHeadSpec, {
    streams, hidden_size, norm_epsilon, epsilon, function, base, scale
}, [function, base, scale]);
record!(routing_intervention::GroupSelectionControl, {
    expected, learned_coefficient_scale, first_row, end_row, row_stride, action, capture_original
}, [action]);

impl MetadataClone for NormalizationScale {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        controls::<Self>()?.checked_add(match self {
            Self::Learned(weight) | Self::LearnedOffset { weight, offset: _ } => {
                weight.metadata_clone_bytes()?
            }
            Self::Unit => 0,
        })
    }
}
impl MetadataClone for GatedProductGroupLayout {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        controls::<Self>()?.checked_add(match self {
            Self::Packed { gate_up, down } => gate_up
                .metadata_clone_bytes()?
                .checked_add(down.metadata_clone_bytes()?)?,
            Self::Independent(groups) => groups.metadata_clone_bytes()?,
        })
    }
}
impl MetadataClone for routing_intervention::GroupSelectionAction {
    fn metadata_clone_bytes(&self) -> Option<usize> {
        controls::<Self>()?.checked_add(match self {
            Self::Exclude(ids) | Self::ZeroContribution(ids) | Self::Force(ids) => {
                ids.metadata_clone_bytes()?
            }
            Self::Bias {
                stage: _,
                ids,
                values,
            } => ids
                .metadata_clone_bytes()?
                .checked_add(values.metadata_clone_bytes()?)?,
        })
    }
}

record!(LinearSpec, { input, output, weight, bias, format }, [weight, bias, format]);
record!(LowRankProjectionSpec, { first, normalization, second }, [first, normalization, second]);

record!(CausalDepthwiseConvolutionSpec, { channels,kernel_size,weight,bias,activation }, [weight,bias]);

record!(EmbeddingSpec, { vocabulary, dimensions, weight, format }, [weight, format]);

// Native adapters may use these exact closed metadata producers without making
// an extra WorkspaceContext. Arbitrary Clone implementations remain private.
impl super::WorkspaceMetadataFunding {
    /// Exact recursive storage of the existing ParameterSpec clone worker.
    pub fn parameter_spec_clone_bytes(source: &ParameterSpec) -> Option<usize> {
        source.metadata_clone_bytes()
    }
    /// Clones one declared parameter after its exact debit. The enclosing
    /// returned value/error must keep this funding alive through retirement.
    pub fn clone_parameter_spec(&self, source: &ParameterSpec) -> Result<ParameterSpec, Error> {
        self.reserve_metadata(Self::parameter_spec_clone_bytes(source)
            .ok_or(WorkspaceMetadataError::Overflow)?).map_err(WorkspaceMetadataError::from)?;
        Ok(source.clone())
    }
    /// Same companion annotation as ordinary parameter binding: source-owned
    /// metadata is cloned, then its weight identity is replaced by this one.
    pub fn clone_parameter_companion(&self, weight: &ParameterSpec, source: &ParameterSpec)
        -> Result<ParameterSpec, Error> {
        let bytes = source.metadata_clone_bytes()
            .and_then(|n| n.checked_add(weight.id.metadata_clone_bytes()?))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.reserve_metadata(bytes).map_err(WorkspaceMetadataError::from)?;
        let mut companion = source.clone();
        companion.linear_companion_of = Some(weight.id.clone());
        Ok(companion)
    }
}

macro_rules! funded_group_spec_clone {
    ($ty:ty,$query:ident,$clone:ident) => {
        impl super::WorkspaceMetadataFunding {
            /// Exact existing recursive metadata clone; grants no native source.
            pub fn $query(source:&$ty)->Option<usize>{source.metadata_clone_bytes()}
            /// Pays the closed clone worker before construction. The caller
            /// retains this account with the returned owned spec and failures.
            pub fn $clone(&self,source:&$ty)->Result<$ty,Error>{
                self.reserve_metadata(Self::$query(source).ok_or(WorkspaceMetadataError::Overflow)?)
                    .map_err(WorkspaceMetadataError::from)?;
                Ok(source.clone())
            }
        }
    }
}
funded_group_spec_clone!(GroupedGatedProductSpec,grouped_gated_product_clone_bytes,clone_grouped_gated_product);
funded_group_spec_clone!(GroupedRelu2Spec,grouped_relu2_clone_bytes,clone_grouped_relu2);
funded_group_spec_clone!(GroupedLinearSpec,grouped_linear_clone_bytes,clone_grouped_linear);
