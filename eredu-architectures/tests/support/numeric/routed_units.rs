fn observe_units(
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    groups: usize,
    mut values: Vec<NumericTensor>,
    observer: &mut dyn GroupedUnitObserver<NumericTensor>,
) -> Result<Vec<NumericTensor>, Error> {
    let tokens = routes.group_indices().shape[0] as usize;
    let top_k = routes.group_indices().shape[1] as usize;
    let width = values[0].data.len();
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by_key(|&row| routes.group_indices().data[row] as usize);
    let sorted = NumericTensor::new(
        vec![order.len() as i32, width as i32],
        order
            .iter()
            .flat_map(|&row| values[row].data.iter().copied())
            .collect(),
    );
    let indices = NumericTensor::new(
        vec![order.len() as i32],
        order.iter().map(|&row| row as f32).collect(),
    );
    let token_indices = NumericTensor::new(
        indices.shape.clone(),
        order.iter().map(|&row| (row / top_k) as f32).collect(),
    );
    let groups_tensor = NumericTensor::new(
        indices.shape.clone(),
        order
            .iter()
            .map(|&row| routes.group_indices().data[row])
            .collect(),
    );
    let batch = GroupedUnitBatch {
        values: &sorted,
        group_indices: &groups_tensor,
        selection_indices: &indices,
        token_indices: &token_indices,
        coefficients: routes.coefficients(),
        token_offset: 0,
        total_token_count: tokens,
        group_count: groups,
    };
    observer.observe(&batch)?;
    let replacement = observer.intervene(&batch)?;
    let effective = replacement.as_ref().unwrap_or(&sorted);
    if effective.shape != sorted.shape {
        return Err(Error::backend_retained_source(
            eredu_nn::GroupedUnitError::ReplacementShape {
                expected: sorted.shape.clone(),
                actual: effective.shape.clone(),
            },
        ));
    }
    observer.observe_effective(&batch.with_values(effective))?;
    for (row, &original) in order.iter().enumerate() {
        values[original] = NumericTensor::new(
            vec![1, width as i32],
            effective.data[row * width..(row + 1) * width].to_vec(),
        );
    }
    assert_eq!(
        input.data.len() / input.shape.last().copied().unwrap() as usize,
        tokens
    );
    Ok(values)
}

pub(super) fn gated(
    bank: &mut NumericExpertBank,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    context: &NumericContext,
    observer: &mut dyn GroupedUnitObserver<NumericTensor>,
    tp: bool,
) -> Result<TensorParallelGroupedOutput<NumericTensor>, Error> {
    if context.bind_checkpoint_values {
        bank.refresh_bound_experts()?;
    }
    let hidden = *input.shape.last().unwrap() as usize;
    let tokens = input.data.len() / hidden;
    let top_k = routes.group_indices().shape[1] as usize;
    let mut values = Vec::new();
    for token in 0..tokens {
        let row = NumericTensor::new(
            vec![1, hidden as i32],
            input.data[token * hidden..(token + 1) * hidden].to_vec(),
        );
        for slot in 0..top_k {
            let expert = &bank.experts[routes.group_indices().data[token * top_k + slot] as usize];
            let gate = linear(&row, &expert.gate, expert.gate_bias.as_ref())?
                .map(|v| bank.policy.gate_upper_bound().map_or(v, |b| v.min(b)));
            let gate = gate.map(|v| match bank.policy.activation() {
                eredu_nn::GatedProductActivation::Silu => {
                    v / (1.0 + (-bank.policy.sigmoid_multiplier() * v).exp())
                }
                eredu_nn::GatedProductActivation::GeluApproximate => {
                    0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044_715 * v.powi(3))).tanh())
                }
                _ => panic!("unsupported fixture activation"),
            });
            let up = linear(&row, &expert.up, expert.up_bias.as_ref())?.map(|v| {
                bank.policy
                    .up_absolute_bound()
                    .map_or(v, |b| v.clamp(-b, b))
                    + bank.policy.up_offset()
            });
            values.push(gate.zip(&up, |a, b| a * b)?);
        }
    }
    let values = observe_units(input, routes, bank.experts.len(), values, observer)?;
    let mut output = NumericTensor::zeros(input.shape.clone());
    let mut bias = bank
        .experts
        .iter()
        .any(|e| e.down_bias.is_some())
        .then(|| NumericTensor::zeros(input.shape.clone()));
    for token in 0..tokens {
        let mut order: Vec<usize> = (0..top_k).collect();
        if bank.spec.reduction() == eredu_nn::GroupReduction::SequentialGroupOrder {
            order.sort_by_key(|&s| routes.group_indices().data[token * top_k + s] as usize);
        }
        for slot in order {
            let index = token * top_k + slot;
            let expert = &bank.experts[routes.group_indices().data[index] as usize];
            let result = linear(
                &values[index],
                &expert.down,
                if tp { None } else { expert.down_bias.as_ref() },
            )?;
            for c in 0..hidden {
                output.data[token * hidden + c] +=
                    routes.coefficients().data[index] * result.data[c];
                if let Some(bias) = &mut bias {
                    bias.data[token * hidden + c] += routes.coefficients().data[index]
                        * expert.down_bias.as_ref().unwrap().data[c];
                }
            }
        }
    }
    Ok(TensorParallelGroupedOutput::new(
        output,
        if tp { bias } else { None },
    ))
}

pub(super) fn relu(
    bank: &mut NumericRelu2Groups,
    input: &NumericTensor,
    routes: &GroupSelection<NumericTensor>,
    observer: &mut dyn GroupedUnitObserver<NumericTensor>,
) -> Result<NumericTensor, Error> {
    let tokens = input.data.len() / bank.hidden;
    let top_k = routes.group_indices().shape[1] as usize;
    let mut values = Vec::new();
    let member = |tensor: &NumericTensor, expert| {
        let value = tensor.axis_slice(0, expert, expert + 1);
        NumericTensor::new(value.shape[1..].to_vec(), value.data)
    };
    for token in 0..tokens {
        let row = NumericTensor::new(
            vec![1, bank.hidden as i32],
            input.data[token * bank.hidden..(token + 1) * bank.hidden].to_vec(),
        );
        for slot in 0..top_k {
            let expert = routes.group_indices().data[token * top_k + slot] as usize;
            values
                .push(linear(&row, &member(&bank.up.0, expert), None)?.map(|v| v.max(0.0).powi(2)));
        }
    }
    let values = observe_units(input, routes, bank.expert_count, values, observer)?;
    let mut output = NumericTensor::zeros(input.shape.clone());
    for token in 0..tokens {
        for slot in 0..top_k {
            let index = token * top_k + slot;
            let expert = routes.group_indices().data[index] as usize;
            let result = linear(&values[index], &member(&bank.down.0, expert), None)?;
            for c in 0..bank.hidden {
                output.data[token * bank.hidden + c] +=
                    routes.coefficients().data[index] * result.data[c];
            }
        }
    }
    Ok(output)
}
