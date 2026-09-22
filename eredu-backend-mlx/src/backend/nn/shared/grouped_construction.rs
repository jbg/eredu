//! One native constructor, with ordinary placeholders or exact acquired rows.
use super::parameters::{
    compact::{self, CompactBindingCause},
    *,
};
use super::*;
use crate::backend::nn::grouped::ParameterFactory;
use eredu_nn::{workspace::HostMetadataFunding, Parameter};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy)]
pub(super) struct Declaration<'a> {
    pub name: &'static str,
    pub source: &'a ParameterSpec,
    pub weight: Option<&'a ParameterSpec>,
}
impl<'a> Declaration<'a> {
    pub fn parameter(name: &'static str, source: &'a ParameterSpec) -> Self {
        Self {
            name,
            source,
            weight: None,
        }
    }
    pub fn companion(
        name: &'static str,
        weight: &'a ParameterSpec,
        source: &'a ParameterSpec,
    ) -> Self {
        Self {
            name,
            source,
            weight: Some(weight),
        }
    }
}
pub(super) enum Constructor<'stream, 'names> {
    Ordinary(&'stream Stream),
    Prepared {
        values: PreparedCompactBindings<'names>,
        funding: HostMetadataFunding,
    },
}
impl<'stream, 'names> Constructor<'stream, 'names> {
    pub fn ordinary(stream: &'stream Stream) -> Self {
        Self::Ordinary(stream)
    }
    /// Host-only constructor frames. Acquired values already own their exact
    /// Slice/Concatenate producer; no scalar, Full or Eval is constructed here.
    pub fn control_bytes<T>(rows: usize) -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<T>(),
            size_of::<Result<T, ComputeError>>(),
            size_of::<[Option<Declaration<'_>>; 8]>(),
            size_of::<Declaration<'_>>(),
            size_of::<[Option<&str>; 4]>(),
            size_of::<[i32; 3]>(),
            size_of::<[i32; 2]>(),
            size_of::<[Option<Parameter<MlxTensor>>; 4]>(),
            size_of::<(Array, Option<Array>, Option<Array>)>(),
            size_of::<(&str, &[i32], Dtype, Option<&[i32]>)>(),
            size_of::<(usize, usize, bool)>(),
            size_of::<Option<HostMetadataFunding>>(),
            size_of::<Result<Option<HostMetadataFunding>, ComputeError>>(),
            crate::backend::nn::grouped::parameter_factory_control_bytes()?,
            compact::linear_control_bytes()?,
            Array::descriptor_comparison_control_bytes()?.checked_mul(rows)?,
            ComputeError::retained_source_construction_bytes::<CompactBindingCause>()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub fn prepared<T>(mut values: PreparedCompactBindings<'names>) -> Result<Self, ComputeError> {
        let controls = Self::control_bytes::<T>(values.len());
        values.begin(controls)?;
        if !values.ready() {
            return Err(values.failure(CompactBindingCause::Identity));
        }
        let funding = values.funding().clone();
        Ok(Self::Prepared { values, funding })
    }
    /// Recursive copies made by the same declared physical parameter rows.
    pub fn parameter_metadata_bytes(
        source: &ParameterSpec,
        weight: Option<&ParameterSpec>,
    ) -> Option<usize> {
        match weight {
            Some(weight) => HostMetadataFunding::parameter_companion_clone_bytes(weight, source),
            None => HostMetadataFunding::parameter_spec_clone_bytes(source),
        }
    }
    pub fn named_metadata_bytes<T>(rows: &[Option<Declaration<'_>>]) -> Option<usize> {
        let count = rows.iter().flatten().count();
        rows.iter().flatten().try_fold(
            Self::control_bytes::<T>(count)?
                .checked_add(NativeParameterTable::control_bytes(count)?)?,
            |bytes, row| bytes.checked_add(Self::parameter_metadata_bytes(row.source, row.weight)?),
        )
    }
    pub fn clone_parameter(
        &self,
        source: &ParameterSpec,
        weight: Option<&ParameterSpec>,
    ) -> Result<ParameterSpec, ComputeError> {
        match self {
            Self::Ordinary(_) => Ok(match weight {
                Some(weight) => bind_linear_companion(weight, source.clone()),
                None => source.clone(),
            }),
            Self::Prepared { funding, .. } => match weight {
                Some(weight) => funding.clone_parameter_companion(weight, source),
                None => funding.clone_parameter_spec(source),
            },
        }
    }
    pub fn finish(self) -> Result<Option<HostMetadataFunding>, ComputeError> {
        match self {
            Self::Ordinary(_) => Ok(None),
            Self::Prepared { values, funding } => {
                if !values.consumed() {
                    return Err(values.failure(CompactBindingCause::Identity));
                }
                Ok(Some(funding))
            }
        }
    }
    pub fn named<M: NativeRetainedValues>(
        self,
        module: M,
        rows: &[Option<Declaration<'_>>],
    ) -> Result<MlxNamedModule<M>, ComputeError> {
        let funding = match &self {
            Self::Ordinary(_) => None,
            Self::Prepared { funding, .. } => Some(funding),
        };
        let mut topology = NativeParameterTable::prepare(rows.iter().flatten().count(), funding)?;
        for row in rows.iter().flatten() {
            topology.push(row.name, self.clone_parameter(row.source, row.weight)?)?;
        }
        let topology = topology.finish()?;
        self.finish()?;
        MlxNamedModule::with_topology(module, topology)
    }
}
impl ParameterFactory for Constructor<'_, '_> {
    type Error = ComputeError;
    fn array(
        &mut self,
        name: &str,
        shape: &[i32],
        dtype: Dtype,
        floating_shape: Option<&[i32]>,
    ) -> Result<Array, ComputeError> {
        match self {
            Self::Ordinary(stream) => compute(safemlx::ops::zeros_dtype(shape, dtype, *stream)),
            Self::Prepared { values, .. } => {
                let value = values
                    .value(name)
                    .ok_or_else(|| values.failure(CompactBindingCause::Identity))?;
                compact::validate_shape(shape, dtype, value, floating_shape)
                    .map_err(|cause| values.failure(cause))?;
                values
                    .take(name)
                    .ok_or_else(|| values.failure(CompactBindingCause::Identity))
            }
        }
    }
    fn geometry(&self, message: &'static str) -> ComputeError {
        match self {
            Self::Ordinary(_) => ComputeError::backend(message),
            Self::Prepared { values, .. } => values.failure(CompactBindingCause::Geometry),
        }
    }
}

use eredu_nn::workspace::ParameterMetadataAllocation;

#[cfg(test)]
mod funding_tests {
    use super::*;
    use eredu_nn::GroupedLinearSpec;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Owner(Arc<AtomicBool>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    fn exact_constructor<T>(
        bytes: usize,
        rows: Vec<(&'static str, Array)>,
        build: impl FnOnce(&HostMetadataFunding, PreparedCompactBindings<'_>) -> Result<T, ComputeError>,
    ) {
        let retired = Arc::new(AtomicBool::new(false));
        let funding = HostMetadataFunding::from_prepaid(
            HostMetadataFunding::prepaid_control_bytes().unwrap()
                + bytes
                + PreparedCompactBindings::layout_bytes(rows.len()).unwrap(),
            eredu_core::HostPreparationAuthority::retain(Owner(retired.clone())),
        )
        .unwrap();
        let mut bindings = PreparedCompactBindings::new(rows.len(), &funding).unwrap();
        for (name, array) in rows {
            bindings.push(name, array).unwrap();
        }
        let model = build(&funding, bindings).unwrap();
        assert!(
            funding.reserve_metadata(1).is_err(),
            "constructor consumed its exact quotation"
        );
        drop(funding);
        assert!(
            !retired.load(Ordering::SeqCst),
            "constructed owner retains its payer"
        );
        drop(model);
        assert!(retired.load(Ordering::SeqCst));
    }
    fn projection(name: &str, bias: bool) -> eredu_nn::GroupedProjectionSpec {
        eredu_nn::GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(),
            bias.then(|| ParameterSpec::trainable(format!("{name}_bias")).unwrap()),
            eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        )
        .unwrap()
    }
    fn array(shape: &[i32]) -> Array {
        let count: usize = shape.iter().map(|&n| usize::try_from(n).unwrap()).product();
        let values: Vec<_> = (0..count).map(|n| (n as f32 + 1.0) / 16.0).collect();
        Array::from_slice(&values, shape)
    }
    #[test]
    fn compact_constructor_quotes_cover_exact_metadata_and_retained_payers() {
        let linear = GroupedLinearSpec::new(
            2,
            2,
            3,
            eredu_nn::GroupedLinearActivation::Identity,
            projection("weight", true),
        )
        .unwrap();
        let bytes = HostMetadataFunding::grouped_linear_clone_bytes(&linear).unwrap()
            + MlxNeuralBackend::grouped_linear_construction_bytes(&linear).unwrap();
        exact_constructor(
            bytes,
            vec![
                ("weight", array(&[2, 3, 2])),
                ("weight_bias", array(&[2, 3])),
            ],
            |funding, bindings| {
                MlxNeuralBackend::grouped_linear_from_bindings(
                    funding.clone_grouped_linear(&linear)?,
                    bindings,
                )
            },
        );

        let gated = GroupedGatedProductSpec::new(
            2,
            2,
            3,
            2,
            eredu_nn::GatedProductPolicy::ordinary_silu(),
            eredu_nn::GatedProductGroupLayout::Packed {
                gate_up: projection("read", true),
                down: projection("write", true),
            },
        )
        .unwrap();
        let bytes = HostMetadataFunding::grouped_gated_product_clone_bytes(&gated).unwrap()
            + MlxNeuralBackend::grouped_gated_product_construction_bytes(&gated)
                .unwrap()
                .unwrap();
        exact_constructor(
            bytes,
            vec![
                ("down_proj", array(&[2, 2, 3])),
                ("down_proj_bias", array(&[2, 2])),
                ("gate_up_proj", array(&[2, 6, 2])),
                ("gate_up_proj_bias", array(&[2, 6])),
            ],
            |funding, bindings| {
                MlxNeuralBackend::grouped_gated_product_from_bindings(
                    funding.clone_grouped_gated_product(&gated)?,
                    bindings,
                )
            },
        );

        let relu = GroupedRelu2Spec::new(
            2,
            2,
            3,
            projection("read", false),
            projection("write", false),
        )
        .unwrap();
        let bytes = HostMetadataFunding::grouped_relu2_clone_bytes(&relu).unwrap()
            + MlxNeuralBackend::grouped_relu2_construction_bytes(&relu).unwrap();
        exact_constructor(
            bytes,
            vec![
                ("down_proj", array(&[2, 2, 3])),
                ("up_proj", array(&[2, 3, 2])),
            ],
            |funding, bindings| {
                MlxNeuralBackend::grouped_relu2_from_bindings(
                    funding.clone_grouped_relu2(&relu)?,
                    bindings,
                )
            },
        );
    }
}
