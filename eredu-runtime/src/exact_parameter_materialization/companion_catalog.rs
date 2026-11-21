//! Exact binding of declared packed parameter slots and their companion tensors.
use super::*;
use eredu_nn::{LinearCompanionRole, ParameterMetadata, ParameterVisitor};

/// Native slot facts for one declared packed linear output.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackedWeightCompanions {
    /// Primary physical weight identity.
    pub weight_name: String,
    /// Scale companion identity.
    pub scales_name: String,
    /// Optional affine bias companion identity.
    pub biases_name: Option<String>,
    /// Physical representation of affine companions.
    pub affine_companion_dtype: RecipeDtype,
}

/// Projects neural parameter metadata and reported storage facts into a companion catalog.
pub fn packed_weight_companions<B, P, M>(
    module: &M,
    quantization: WeightQuantization,
) -> Result<BTreeMap<String, PackedWeightCompanions>, BackendFailure>
where
    B: NeuralBackend,
    P: ExactParameterMaterializer<B>,
    M: Parameterized<B::Tensor>,
{
    struct Collector {
        parameters: BTreeMap<String, RecipeDtype>,
        companions: BTreeMap<(String, LinearCompanionRole), (String, RecipeDtype)>,
        error: Option<BackendFailure>,
    }

    struct Inspect<B, P>(Collector, std::marker::PhantomData<fn() -> (B, P)>);

    impl<'a, B, P> ParameterVisitor<'a, B::Tensor> for Inspect<B, P>
    where
        B: NeuralBackend,
        P: ExactParameterMaterializer<B>,
    {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a B::Tensor) {
            let collector = &mut self.0;
            if collector.error.is_some() {
                return;
            }
            let name = metadata.id.as_str().to_owned();
            let Some(slot) = P::parameter_target(value) else {
                collector.error = Some(invalid(format!(
                    "parameter {name:?} has no native geometry"
                )));
                return;
            };
            collector
                .parameters
                .insert(name.clone(), slot.dtype.clone());
            if let Some(role) = metadata.linear_companion {
                let Some(weight) = metadata.linear_companion_of else {
                    collector.error = Some(invalid(format!(
                        "linear quantization companion {name:?} has no primary weight identity"
                    )));
                    return;
                };
                if collector
                    .companions
                    .insert(
                        (weight.as_str().to_owned(), role),
                        (name.clone(), slot.dtype.clone()),
                    )
                    .is_some()
                {
                    collector.error = Some(invalid(format!(
                        "linear weight {:?} declares more than one {role:?} companion",
                        weight.as_str()
                    )));
                }
            }
        }
    }

    let collector = Collector {
        parameters: BTreeMap::new(),
        companions: BTreeMap::new(),
        error: None,
    };
    let mut inspection = Inspect::<B, P>(collector, std::marker::PhantomData);
    module.visit_parameters(&mut inspection);
    let Inspect(mut collector, _) = inspection;
    if let Some(error) = collector.error {
        return Err(error);
    }
    let weights = collector
        .companions
        .keys()
        .map(|(weight, _)| weight.clone())
        .collect::<BTreeSet<_>>();
    weights
        .into_iter()
        .map(|weight_name| {
            let dtype = collector.parameters.get(&weight_name).ok_or_else(|| {
                invalid(format!(
                    "linear quantization companions reference missing weight {weight_name:?}"
                ))
            })?;
            if *dtype != RecipeDtype::U32 {
                return Ok(None);
            }
            let (scales_name, scales_dtype) = collector
                .companions
                .remove(&(weight_name.clone(), LinearCompanionRole::Scale))
                .ok_or_else(|| {
                    invalid(format!(
                        "packed linear weight {weight_name:?} has no declared scale companion"
                    ))
                })?;
            let biases = collector
                .companions
                .remove(&(weight_name.clone(), LinearCompanionRole::AffineBias));
            if quantization.has_biases() && biases.is_none() {
                return Err(invalid(format!(
                    "affine packed linear weight {weight_name:?} has no declared bias companion"
                )));
            }
            let affine_companion_dtype = match scales_dtype {
                RecipeDtype::F16 => RecipeDtype::F16,
                RecipeDtype::BF16 => RecipeDtype::BF16,
                RecipeDtype::F32 => RecipeDtype::F32,
                RecipeDtype::U8 if !quantization.has_biases() => RecipeDtype::F32,
                dtype => {
                    return Err(invalid(format!(
                        "packed linear weight {weight_name:?} has unsupported scale dtype {dtype:?}"
                    )))
                }
            };
            if let Some((_, biases_dtype)) = &biases {
                let expected = match affine_companion_dtype {
                    RecipeDtype::F16 => RecipeDtype::F16,
                    RecipeDtype::BF16 => RecipeDtype::BF16,
                    RecipeDtype::F32 => RecipeDtype::F32,
                    _ => unreachable!("selected affine companion dtype"),
                };
                if *biases_dtype != expected {
                    return Err(invalid(format!(
                        "packed linear weight {weight_name:?} has mismatched scale and bias dtypes"
                    )));
                }
            }
            Ok(Some((
                weight_name.clone(),
                PackedWeightCompanions {
                    weight_name,
                    scales_name,
                    biases_name: biases.map(|(name, _)| name),
                    affine_companion_dtype,
                },
            )))
        })
        .collect::<Result<Vec<_>, BackendFailure>>()
        .map(|targets| targets.into_iter().flatten().collect())
}

pub(crate) fn bind_packed_transform_targets<B, P, SM, U>(
    target_static: &SM,
    target_units: &[U],
    quantization: WeightQuantization,
    targets: Vec<BoundedQuantizationTarget>,
) -> Result<Vec<BoundedQuantizationTarget>, BackendFailure>
where
    B: NeuralBackend,
    P: ExactParameterMaterializer<B>,
    SM: Parameterized<B::Tensor>,
    U: Parameterized<B::Tensor>,
{
    let mut available = packed_weight_companions::<B, P, _>(target_static, quantization)?;
    for unit in target_units {
        for (target, companions) in packed_weight_companions::<B, P, _>(unit, quantization)? {
            if available.insert(target.clone(), companions).is_some() {
                return Err(invalid(format!(
                    "selected packed target {target:?} appears in more than one module"
                )));
            }
        }
    }
    let targets = targets
        .into_iter()
        .map(|target| {
            let name = target.weight_name();
            let companions = available.remove(name).ok_or_else(|| {
                invalid(format!(
                    "selected transform {name:?} has no native packed target"
                ))
            })?;
            if target.scales_name() != companions.scales_name
                || target.biases_name() != companions.biases_name.as_deref()
            {
                return Err(invalid(format!(
                    "selected transform {name:?} companion identities differ from native target"
                )));
            }
            match quantization {
                WeightQuantization::Affine(_) => target
                    .with_affine_companion_dtype(companions.affine_companion_dtype)
                    .map_err(BackendFailure::from_error),
                WeightQuantization::MxFp4 => Ok(target),
                _ => Err(invalid(format!(
                    "load-time encoding for {quantization:?} is unavailable"
                ))),
            }
        })
        .collect::<Result<Vec<_>, BackendFailure>>()?;
    if !available.is_empty() {
        return Err(invalid(format!(
            "native packed modules contain unselected targets: {:?}",
            available.keys().collect::<Vec<_>>()
        )));
    }
    Ok(targets)
}
