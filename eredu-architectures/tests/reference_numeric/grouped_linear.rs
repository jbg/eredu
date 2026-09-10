use super::*;
use eredu_nn::{GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec};

pub(super) fn addressable_bank(
    plan: &ExpertRealizationPlan<GroupedLinearSpec>,
    catalog: &ExpertResidencyCatalog,
    selected: &[eredu_runtime::AddressableBankMember],
    capacity: usize,
    context: &NumericContext,
) -> Result<NumericGroupedBankMechanism, String> {
    let mut full_banks = BTreeMap::new();
    let mut linear = BTreeMap::new();
    let mut bytes = BTreeMap::new();
    for unit in catalog.units() {
        let address = (
            unit.owner_group().as_str().to_owned(),
            unit.identity().unit(),
        );
        if !full_banks.contains_key(&address) {
            let spec = plan
                .unit_spec(&address.0, address.1)
                .ok_or_else(|| format!("linear unit {address:?} has no plan"))?;
            full_banks.insert(
                address.clone(),
                NumericLinearGroups::new(spec.clone(), context)
                    .map_err(|error| error.to_string())?,
            );
        }
        let full = &full_banks[&address];
        let spec = full
            .spec
            .clone()
            .with_group_count(1)
            .map_err(|error| error.to_string())?;
        linear.insert(
            unit.identity(),
            full.selected(&spec, &[unit.identity().member()])
                .map_err(|error| error.to_string())?,
        );
        bytes.insert(
            unit.identity(),
            selected
                .iter()
                .find(|member| member.key() == unit.identity())
                .map(eredu_runtime::AddressableBankMember::selected_bytes)
                .or(unit.byte_len())
                .ok_or_else(|| "linear unit has no admitted byte geometry".to_owned())?,
        );
    }
    Ok(NumericGroupedBankMechanism {
        banks: BTreeMap::new(),
        linear,
        bytes,
        report: NumericBankReport::default(),
        resident: Vec::new(),
        capacity,
    })
}

#[derive(Debug, Clone)]
pub(super) struct NumericLinearGroups {
    spec: GroupedLinearSpec,
    weight: NumericTensor,
    metadata: ParameterMetadata,
    bias: Option<(ParameterMetadata, NumericTensor)>,
}

impl NumericLinearGroups {
    pub(super) fn from_bound_weight(
        spec: GroupedLinearSpec,
        weight: NumericTensor,
    ) -> Result<Self, Error> {
        if weight.shape
            != [
                spec.group_count(),
                spec.output_dimensions(),
                spec.input_dimensions(),
            ]
        {
            return Err(Error::backend("scalar member linear shape mismatch"));
        }
        let metadata = ParameterMetadata::from_spec(spec.projection().weight(), false);
        Ok(Self {
            spec,
            weight,
            metadata,
            bias: None,
        })
    }

    pub(super) fn concatenate(
        banks: &[Self],
        spec: &GroupedLinearSpec,
        context: &NumericContext,
    ) -> Result<Self, Error> {
        let mut bank = banks
            .first()
            .cloned()
            .ok_or_else(|| Error::backend("empty linear acquisition"))?;
        if banks.len() != spec.group_count() as usize {
            return Err(Error::backend(
                "linear acquisition count differs from compact spec",
            ));
        }
        bank.spec = spec.clone();
        bank.weight = NumericTensor::concatenate(
            &banks
                .iter()
                .map(|bank| bank.weight.clone())
                .collect::<Vec<_>>(),
            0,
            context,
        )?;
        if let Some((_, bias)) = &mut bank.bias {
            *bias = NumericTensor::concatenate(
                &banks
                    .iter()
                    .map(|bank| bank.bias.as_ref().unwrap().1.clone())
                    .collect::<Vec<_>>(),
                0,
                context,
            )?;
        }
        Ok(bank)
    }

    pub(super) fn new(spec: GroupedLinearSpec, context: &NumericContext) -> Result<Self, Error> {
        spec.validate()?;
        let projection = spec.projection();
        let weight = local_parameter(
            projection.weight(),
            vec![
                spec.group_count(),
                spec.output_dimensions(),
                spec.input_dimensions(),
            ],
            false,
            context,
        )?;
        let metadata =
            ParameterMetadata::from_spec(projection.weight(), projection.weight().trainable);
        let bias = projection
            .bias()
            .map(|bias| {
                Ok::<_, Error>((
                    ParameterMetadata::from_spec(bias, bias.trainable),
                    local_parameter(
                        bias,
                        vec![spec.group_count(), spec.output_dimensions()],
                        false,
                        context,
                    )?,
                ))
            })
            .transpose()?;
        Ok(Self {
            spec,
            weight,
            metadata,
            bias,
        })
    }
}

impl Parameterized<NumericTensor> for NumericLinearGroups {
    fn visit_parameters<'a, V: ParameterVisitor<'a, NumericTensor>>(&'a self, visitor: &mut V) {
        visit(&self.metadata, &self.weight, visitor);
        if let Some((metadata, bias)) = &self.bias {
            visit(metadata, bias, visitor);
        }
    }
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, NumericTensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        visit_mut(&self.metadata, &mut self.weight, visitor);
        if let Some((metadata, bias)) = &mut self.bias {
            visit_mut(metadata, bias, visitor);
        }
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
        if let Some((metadata, _)) = &mut self.bias {
            metadata.trainable = trainable;
        }
    }
}

impl GroupedLinearOperator<NumericTensor> for NumericLinearGroups {
    fn spec(&self) -> &GroupedLinearSpec {
        &self.spec
    }
    fn forward_grouped(
        &mut self,
        input: &NumericTensor,
        routes: &GroupSelection<NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let width = self.spec.input_dimensions() as usize;
        let out = self.spec.output_dimensions() as usize;
        if input.shape.len() != 2
            || input.shape[1] as usize != width
            || routes.group_indices().shape.len() != 2
            || routes.group_indices().shape[0] != input.shape[0]
            || routes.group_indices().shape != routes.coefficients().shape
        {
            return Err(Error::backend("selected linear geometry mismatch"));
        }
        let top = routes.group_indices().shape[1] as usize;
        let mut output = NumericTensor::zeros(vec![input.shape[0], out as i32]);
        for row in 0..input.shape[0] as usize {
            let mut order = (0..top).collect::<Vec<_>>();
            if self.spec.reduction() == eredu_nn::GroupReduction::SequentialGroupOrder {
                order.sort_by_key(|&selected| {
                    routes.group_indices().data[row * top + selected] as usize
                });
            }
            for selected in order {
                let id = routes.group_indices().data[row * top + selected];
                if id < 0.0 || id.fract() != 0.0 || id >= self.spec.group_count() as f32 {
                    return Err(Error::backend("selected linear expert ID is invalid"));
                }
                let id = id as usize;
                for column in 0..out {
                    let mut value = (0..width)
                        .map(|k| {
                            input.data[row * width + k]
                                * self.weight.data[(id * out + column) * width + k]
                        })
                        .sum::<f32>();
                    if let Some((_, bias)) = &self.bias {
                        value += bias.data[id * out + column];
                    }
                    if self.spec.activation() == GroupedLinearActivation::Silu {
                        value /= 1.0 + (-value).exp();
                    }
                    output.data[row * out + column] +=
                        value * routes.coefficients().data[row * top + selected];
                }
            }
        }
        Ok(output)
    }
}

impl NumericLinearGroups {
    pub(super) fn selected(&self, spec: &GroupedLinearSpec, ids: &[usize]) -> Result<Self, Error> {
        let mut selected = self.clone();
        selected.spec = spec.clone();
        let columns =
            self.spec.output_dimensions() as usize * self.spec.input_dimensions() as usize;
        let values = ids
            .iter()
            .flat_map(|id| {
                self.weight.data[id * columns..(id + 1) * columns]
                    .iter()
                    .copied()
            })
            .collect();
        selected.weight = NumericTensor::new(
            vec![
                ids.len() as i32,
                self.spec.output_dimensions(),
                self.spec.input_dimensions(),
            ],
            values,
        );
        if let Some((_, bias)) = &mut selected.bias {
            let width = self.spec.output_dimensions() as usize;
            *bias = NumericTensor::new(
                vec![ids.len() as i32, width as i32],
                ids.iter()
                    .flat_map(|id| {
                        self.bias.as_ref().unwrap().1.data[id * width..(id + 1) * width]
                            .iter()
                            .copied()
                    })
                    .collect(),
            );
        }
        Ok(selected)
    }
}
