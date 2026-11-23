//! Local selection and execution for resident grouped projections.

use candle_core::{DType, Tensor as NativeTensor};
use eredu_nn::{
    Error, GatedProductGroupLayout, GroupReduction, GroupScoring, GroupSelection,
    GroupSelectionOperator, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedLinearOperator, GroupedLinearSpec, GroupedNeuralBackend, GroupedProjectionSpec,
    GroupedRelu2Operator, GroupedRelu2Spec, GroupedUnitBatch, GroupedUnitObserver,
    JointGroupSelection, JointGroupSelectionInput, LinearOperator, LinearSpec, Parameter,
    Parameterized, RoutingPrecision, Tensor, TopKGroupSelectorSpec,
};

use crate::{operators, quantized, CandleContext, CandleLinear, CandleNeuralBackend, CandleTensor};

fn unavailable() -> Error {
    Error::backend("requested grouped mechanism is unavailable on this backend")
}

fn round_float(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_f32(value).to_f32(),
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        _ => value,
    }
}

fn routing_dtype(precision: RoutingPrecision, input: DType, current: DType) -> DType {
    match precision {
        RoutingPrecision::Preserve => current,
        RoutingPrecision::Input => input,
        RoutingPrecision::Float32 => DType::F32,
    }
}

/// Learned local selector with explicit score, ranking, and coefficient stages.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleSelector {
    projection: CandleLinear,
    correction: Option<Parameter<CandleTensor>>,
    input_scale: Option<Parameter<CandleTensor>>,
    coefficient_scale: Option<Parameter<CandleTensor>>,
    #[parameter(skip)]
    spec: TopKGroupSelectorSpec,
}

impl CandleSelector {
    fn project_logits(
        &mut self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let dtype = input.value.dtype();
        let width = self.spec.input_dimensions() as usize;
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
            || input.value.dims().last() != Some(&width)
        {
            return Err(Error::backend("invalid router input precision or width"));
        }
        let flat = CandleTensor::wrap(
            input
                .value
                .reshape((input.value.elem_count() / width, width)),
        )?;
        let mut transformed = flat;
        if let Some(transform) = self.spec.input_transform() {
            let normalized =
                operators::rms_norm(&transformed, None, None, None, transform.epsilon())?;
            transformed = normalized.multiply(
                self.input_scale
                    .as_ref()
                    .expect("declared input scale")
                    .as_ref(),
                context,
            )?;
            if transform.inverse_sqrt_dimensions() {
                let scale = round_float((width as f32).sqrt().recip(), transformed.value.dtype());
                transformed = CandleTensor::wrap(transformed.value.affine(scale.into(), 0.))?;
            }
        }
        let precision = self.spec.arithmetic().projection;
        if precision == RoutingPrecision::Float32 {
            transformed = CandleTensor::wrap(transformed.value.to_dtype(DType::F32))?;
        }
        let projected = self.projection.forward(&transformed, context)?;
        let dtype = routing_dtype(precision, dtype, projected.value.dtype());
        CandleTensor::wrap(projected.value.to_dtype(dtype).and_then(|value| {
            if self.spec.logit_scale() == 1. {
                Ok(value)
            } else {
                value.affine(round_float(self.spec.logit_scale(), dtype).into(), 0.)
            }
        }))
    }

    fn score(
        &self,
        projected: &CandleTensor,
        input: DType,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let dtype = routing_dtype(
            self.spec.arithmetic().scores,
            input,
            projected.value.dtype(),
        );
        let projected = CandleTensor::wrap(projected.value.to_dtype(dtype))?;
        match self.spec.selection().scoring() {
            GroupScoring::Softmax => Self::softmax(&projected),
            GroupScoring::SelectedSoftmax => Ok(projected),
            GroupScoring::Sigmoid | GroupScoring::SqrtSoftplus => {
                let values = projected
                    .to_f32_vec(context)?
                    .into_iter()
                    .map(|x| {
                        if self.spec.selection().scoring() == GroupScoring::Sigmoid {
                            eredu_nn::routing_selection::sigmoid(x)
                        } else {
                            round_float(eredu_nn::routing_selection::softplus(x), dtype).sqrt()
                        }
                    })
                    .collect();
                CandleTensor::wrap(
                    NativeTensor::from_vec(values, projected.value.shape(), &context.device)
                        .and_then(|value| value.to_dtype(dtype)),
                )
            }
            _ => Err(unavailable()),
        }
    }

    fn ranking(
        &self,
        scores: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        match &self.correction {
            Some(bias) => scores.add(bias.as_ref(), context),
            None => Ok(scores.clone()),
        }
    }

    fn softmax(projected: &CandleTensor) -> Result<CandleTensor, Error> {
        let dtype = projected.value.dtype();
        let wide = projected
            .value
            .to_dtype(DType::F32)
            .map_err(Error::backend_source)?;
        let scores = if crate::float_cpu::applies(&wide) {
            crate::float_cpu::softmax(&wide)
        } else {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if crate::metal_f32::applies(&wide) {
                crate::metal_f32::attention_softmax(&wide)
            } else {
                candle_nn::ops::softmax(&wide, candle_core::D::Minus1)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            candle_nn::ops::softmax(&wide, candle_core::D::Minus1)
        };
        CandleTensor::wrap(scores.and_then(|scores| scores.to_dtype(dtype)))
    }

    fn output_shape(&self, scores: &CandleTensor) -> Result<[usize; 2], Error> {
        let width = self.spec.selection().group_count() as usize;
        if scores.value.dims().last() != Some(&width) {
            return Err(Error::backend("router scores have the wrong expert width"));
        }
        Ok([
            scores.value.elem_count() / width,
            self.spec.selection().top_k() as usize,
        ])
    }

    fn rank(&self, scores: &CandleTensor, context: &CandleContext) -> Result<CandleTensor, Error> {
        let output_shape = self.output_shape(scores)?;
        let ids = self.rank_rows(&scores.to_f32_vec(context)?, scores.value.dtype())?;
        CandleTensor::wrap(NativeTensor::from_vec(
            ids,
            output_shape.as_slice(),
            &context.device,
        ))
    }

    fn rank_rows(&self, scores: &[f32], dtype: DType) -> Result<Vec<i32>, Error> {
        let spec = self.spec.selection();
        let width = spec.group_count() as usize;
        let mut ids = Vec::with_capacity(scores.len() / width * spec.top_k() as usize);
        for row in scores.chunks_exact(width) {
            ids.extend(eredu_nn::routing_selection::select_row(row, spec, |sum| {
                round_float(sum, dtype)
            })?);
        }
        Ok(ids)
    }

    fn weights(
        &self,
        scores: &CandleTensor,
        ids: CandleTensor,
        input_dtype: DType,
        context: &CandleContext,
    ) -> Result<GroupSelection<CandleTensor>, Error> {
        let output_shape = self.output_shape(scores)?;
        let [rows, slots] = output_shape;
        if ids.value.dims().last() != Some(&slots) || ids.value.elem_count() != rows * slots {
            return Err(Error::backend(
                "group indices do not match the selector shape",
            ));
        }
        self.weight_rows(
            scores,
            &scores.to_f32_vec(context)?,
            ids.to_i32_vec(context)?,
            input_dtype,
            context,
        )
    }

    fn weight_rows(
        &self,
        scores: &CandleTensor,
        data: &[f32],
        selected_ids: Vec<i32>,
        input_dtype: DType,
        context: &CandleContext,
    ) -> Result<GroupSelection<CandleTensor>, Error> {
        let output_shape = self.output_shape(scores)?;
        let dtype = scores.value.dtype();
        let spec = self.spec.selection();
        let groups = spec.group_count() as usize;
        let slots = spec.top_k() as usize;
        let mut selected_scores = Vec::with_capacity(selected_ids.len());
        for (row, selected) in data
            .chunks_exact(groups)
            .zip(selected_ids.chunks_exact(slots))
        {
            for &id in selected {
                let score = row
                    .get(id as usize)
                    .filter(|score| score.is_finite())
                    .ok_or_else(|| Error::backend("invalid selected group or score"))?;
                selected_scores.push(*score);
            }
        }
        if spec.scoring() == GroupScoring::SelectedSoftmax {
            let gathered = CandleTensor::wrap(
                NativeTensor::from_vec(selected_scores, &output_shape[..], &context.device)
                    .and_then(|value| value.to_dtype(dtype)),
            )?;
            selected_scores = Self::softmax(&gathered)?.to_f32_vec(context)?;
        }
        let mut coefficients = Vec::with_capacity(selected_ids.len());
        for row in selected_scores.chunks_exact(slots) {
            let denominator = if spec.normalize_selected() {
                let sum = if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
                    crate::float_cpu::sum_row(row)
                } else {
                    row.iter().map(|&score| f64::from(score)).sum::<f64>() as f32
                };
                round_float(
                    round_float(sum, dtype) + round_float(spec.normalization_epsilon(), dtype),
                    dtype,
                )
            } else {
                1.
            };
            if !denominator.is_finite() || denominator <= 0. {
                return Err(Error::backend("invalid group selection normalization"));
            }
            coefficients.extend(row.iter().map(|&score| {
                let normalized = round_float(score / denominator, dtype);
                round_float(
                    normalized * round_float(spec.coefficient_scale(), dtype),
                    dtype,
                )
            }));
        }
        let materialize = |values, dtype| {
            CandleTensor::wrap(
                NativeTensor::from_vec(values, &output_shape[..], &context.device)
                    .and_then(|tensor| tensor.to_dtype(dtype)),
            )
        };
        let ids = CandleTensor::wrap(NativeTensor::from_vec(
            selected_ids,
            output_shape.as_slice(),
            &context.device,
        ))?;
        let coefficients = materialize(coefficients, dtype)?;
        let coefficients = match &self.coefficient_scale {
            Some(scale) => {
                coefficients.multiply(&scale.as_ref().take_axis(&ids, 0, context)?, context)?
            }
            None => coefficients,
        };
        let coefficient_dtype = routing_dtype(
            self.spec.arithmetic().coefficients,
            input_dtype,
            coefficients.value.dtype(),
        );
        Ok(GroupSelection::new(
            ids,
            materialize(selected_scores, dtype)?,
            CandleTensor::wrap(coefficients.value.to_dtype(coefficient_dtype))?,
        ))
    }

    fn select_impl(
        &mut self,
        input: &CandleTensor,
        indices: Option<&CandleTensor>,
        context: &CandleContext,
    ) -> Result<GroupSelection<CandleTensor>, Error> {
        let dtype = input.value.dtype();
        let raw = self.project_logits(input, context)?;
        let scores = self.score(&raw, dtype, context)?;
        let ids = match indices {
            Some(ids) => ids.clone(),
            None => self.rank(&self.ranking(&scores, context)?, context)?,
        };
        self.weights(&scores, ids, dtype, context)
    }
}

mod routing;

impl GroupSelectionOperator<CandleTensor> for CandleSelector {
    fn select_intervened(
        &mut self,
        input: &CandleTensor,
        control: &eredu_nn::routing_intervention::GroupSelectionControl,
        context: &CandleContext,
    ) -> Result<eredu_nn::routing_intervention::IntervenedGroupSelection<CandleTensor>, Error> {
        eredu_nn::routing_intervention::execute_routing_intervention(
            &mut routing::NativeRouting {
                selector: self,
                context,
                input_dtype: input.value.dtype(),
            },
            input,
            control,
        )
        .map_err(Error::backend_source)
    }

    fn select(
        &mut self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<GroupSelection<CandleTensor>, Error> {
        self.select_impl(input, None, context)
    }

    fn select_indices(
        &mut self,
        input: &CandleTensor,
        ids: &CandleTensor,
        context: &CandleContext,
    ) -> Result<GroupSelection<CandleTensor>, Error> {
        self.select_impl(input, Some(ids), context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
struct GroupedProjection {
    weight: Parameter<CandleTensor>,
    bias: Option<Parameter<CandleTensor>>,
    #[parameter(skip)]
    packed: Option<quantized::PackedMatrix>,
    words: Option<crate::word_projection::WordProjection>,
    block: Option<crate::block_projection::BlockProjection>,
}

impl GroupedProjection {
    fn new(
        spec: &GroupedProjectionSpec,
        groups: Option<i32>,
        rows: i32,
        columns: i32,
        context: &CandleContext,
    ) -> Result<Self, Error> {
        if spec.format().arithmetic() != eredu_nn::LinearArithmetic::Native {
            return Err(Error::backend(
                "grouped projection requires native arithmetic",
            ));
        }
        let (matrix, packed) = quantized::matrix_slot(spec.format(), rows, columns, context)?;
        let words = crate::word_projection::WordProjection::new(
            spec.format(),
            rows,
            columns,
            groups,
            context,
        )?;
        let block = crate::block_projection::BlockProjection::new(
            spec.format(),
            rows,
            columns,
            groups,
            context,
        )?;
        let slot = if let Some(groups) = groups {
            let shape = matrix.value.dims();
            CandleTensor::wrap(
                NativeTensor::zeros((), matrix.value.dtype(), &context.device)
                    .and_then(|zero| zero.broadcast_as((groups as usize, shape[0], shape[1]))),
            )?
        } else {
            matrix
        };
        let bias = spec
            .bias()
            .map(|parameter| {
                let shape = match groups {
                    Some(groups) => vec![groups, rows],
                    None => vec![rows],
                };
                CandleTensor::unloaded_f32(&shape, context)
                    .map(|value| Parameter::new(parameter.clone(), value))
            })
            .transpose()?;
        Ok(Self {
            weight: Parameter::new(spec.weight().clone(), slot),
            bias,
            packed,
            words,
            block,
        })
    }

    fn project(
        &self,
        group: Option<usize>,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let slice = |value: &CandleTensor| match group {
            Some(group) => {
                CandleTensor::wrap(value.value.narrow(0, group, 1).and_then(|x| x.squeeze(0)))
            }
            None => Ok(value.clone()),
        };
        let weight = slice(self.weight.as_ref())?;
        let bias = self
            .bias
            .as_ref()
            .map(|value| slice(value.as_ref()))
            .transpose()?;
        if let Some(block) = &self.block {
            let activation = block.input(input, context)?;
            return block.project(
                input,
                activation.as_ref(),
                &weight,
                bias.as_ref(),
                self.packed.as_ref(),
                group,
            );
        }
        if let Some(words) = &self.words {
            return words.project(input, &weight, bias.as_ref(), group);
        }
        match &self.packed {
            Some(packed) => packed.grouped_linear(input, &weight, bias.as_ref()),
            None => CandleTensor::linear(input, &weight, bias.as_ref(), context),
        }
    }
}

/// Gated-product bank with resident, selectable parameter groups.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleGatedProductGroups {
    packed: Option<PackedGatedProduct>,
    independent: Vec<IndependentGatedProduct>,
    #[parameter(skip)]
    spec: GroupedGatedProductSpec,
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
struct PackedGatedProduct {
    gate_up: GroupedProjection,
    down: GroupedProjection,
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
struct IndependentGatedProduct {
    gate: GroupedProjection,
    up: GroupedProjection,
    down: GroupedProjection,
}

struct GroupedRows {
    tokens: usize,
    slots: usize,
    ids: Vec<i32>,
    outputs: Vec<f32>,
}

#[derive(Clone, Copy)]
struct GroupGeometry {
    input: usize,
    output: usize,
    groups: usize,
    reduction: GroupReduction,
}

impl GroupGeometry {
    fn project_slots(
        self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
        observer: Option<&mut dyn GroupedUnitObserver<CandleTensor>>,
        activate: impl Fn(usize, &CandleTensor) -> Result<CandleTensor, Error>,
        finish: impl Fn(usize, CandleTensor) -> Result<CandleTensor, Error>,
    ) -> Result<GroupedRows, Error> {
        let input_width = self.input;
        let output_width = self.output;
        let shape = input.value.dims();
        let ids = &selections.group_indices().value;
        let coefficients = &selections.coefficients().value;
        let (tokens, slots) = ids.dims2().map_err(Error::backend_source)?;
        if shape.len() < 2
            || shape.last() != Some(&input_width)
            || tokens != input.value.elem_count() / input_width
            || ids.dims() != coefficients.dims()
        {
            return Err(Error::backend("invalid grouped input or selection shape"));
        }
        if slots == 0 {
            return Err(Error::backend(
                "grouped execution requires at least one slot",
            ));
        }
        let ids = selections.group_indices().to_i32_vec(context)?;
        let groups = self.groups;
        let mut buckets = vec![Vec::new(); groups];
        for (slot, &id) in ids.iter().enumerate() {
            buckets
                .get_mut(id as usize)
                .ok_or_else(|| Error::backend("group ID is outside the packed bank"))?
                .push(slot);
        }
        let input_rows = input
            .value
            .reshape((tokens, input_width))
            .map_err(Error::backend_source)?;
        let activated = |group, routes: &[usize]| {
            let rows = routes
                .iter()
                .map(|slot| u32::try_from(slot / slots))
                .collect::<Result<Vec<_>, _>>()
                .map_err(Error::backend_source)?;
            let rows = CandleTensor::wrap(
                NativeTensor::new(rows, &context.device)
                    .and_then(|indices| input_rows.index_select(&indices, 0)),
            )?;
            activate(group, &rows)
        };
        let active = || {
            buckets
                .iter()
                .enumerate()
                .filter(|(_, routes)| !routes.is_empty())
        };
        // Collect only actual routed rows, once, at the activated-unit boundary.
        // An ordinary invocation retains only one group's activated tensor.
        let observed = if let Some(observer) = observer.filter(|_| !ids.is_empty()) {
            let values = active()
                .map(|(group, routes)| activated(group, routes))
                .collect::<Result<Vec<_>, _>>()?;
            let values = CandleTensor::concatenate(&values, 0, context)?;
            let route_order = active()
                .flat_map(|(_, routes)| routes.iter().copied())
                .collect::<Vec<_>>();
            let indices = |indices: Vec<usize>| {
                let indices = indices
                    .into_iter()
                    .map(i32::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(Error::backend_source)?;
                let count = i32::try_from(indices.len()).map_err(Error::backend_source)?;
                CandleTensor::from_i32_slice(&indices, &[count], context)
            };
            let token_indices = indices(route_order.iter().map(|slot| slot / slots).collect())?;
            let group_indices = CandleTensor::from_i32_slice(
                &route_order
                    .iter()
                    .map(|slot| ids[*slot])
                    .collect::<Vec<_>>(),
                &[i32::try_from(route_order.len()).map_err(Error::backend_source)?],
                context,
            )?;
            let selection_indices = indices(route_order)?;
            let coefficients = CandleTensor::wrap(coefficients.reshape((tokens, slots)))?;
            let batch = GroupedUnitBatch {
                values: &values,
                group_indices: &group_indices,
                selection_indices: &selection_indices,
                token_indices: &token_indices,
                coefficients: &coefficients,
                token_offset: 0,
                total_token_count: tokens,
                group_count: groups,
            };
            Some(batch.apply_observer(observer, |a, b| a.value.dtype() == b.value.dtype())?)
        } else {
            None
        };
        let mut outputs = vec![
            0.;
            ids.len()
                .checked_mul(output_width)
                .ok_or_else(unavailable)?
        ];
        let mut offset = 0;
        for (group, routes) in active() {
            let product = match &observed {
                Some(values) => CandleTensor::wrap(values.value.narrow(0, offset, routes.len()))?,
                None => activated(group, routes)?,
            };
            offset += routes.len();
            let rows = finish(group, product)?;
            if rows.value.dims() != [routes.len(), output_width] {
                return Err(Error::backend(
                    "group projection output differs from its declared geometry",
                ));
            }
            let rows = rows.to_f32_vec(context)?;
            for (&slot, row) in routes.iter().zip(rows.chunks_exact(output_width)) {
                outputs[slot * output_width..][..output_width].copy_from_slice(row);
            }
        }
        Ok(GroupedRows {
            tokens,
            slots,
            ids,
            outputs,
        })
    }

    fn reduce_rows(
        &self,
        rows: GroupedRows,
        selections: &GroupSelection<CandleTensor>,
        shape: &[usize],
        dtype: DType,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let GroupedRows {
            tokens,
            slots,
            ids,
            outputs,
        } = rows;
        let output_width = self.output;
        let sequential = self.reduction == GroupReduction::SequentialGroupOrder;
        let coefficients = selections.coefficients().to_f32_vec(context)?;
        if coefficients.len() != ids.len() {
            return Err(Error::backend(
                "grouped output and coefficients differ in route count",
            ));
        }
        let mut output = vec![0.; tokens.checked_mul(output_width).ok_or_else(unavailable)?];
        let mut order: Vec<_> = (0..slots).collect();
        for (token, out) in output.chunks_exact_mut(output_width).enumerate() {
            if sequential {
                order.sort_unstable_by_key(|&slot| (ids[token * slots + slot], slot));
            }
            for &slot in &order {
                let slot = token * slots + slot;
                let offset = slot * output_width;
                for (out, value) in out.iter_mut().zip(&outputs[offset..][..output_width]) {
                    let weighted = coefficients[slot] * value;
                    *out = if sequential {
                        round_float(*out + round_float(weighted, dtype), dtype)
                    } else {
                        *out + weighted
                    };
                }
            }
        }
        CandleTensor::wrap(
            NativeTensor::from_vec(output, shape, &context.device)
                .and_then(|value| value.to_dtype(dtype)),
        )
    }
}

impl CandleGatedProductGroups {
    fn geometry(&self) -> GroupGeometry {
        GroupGeometry {
            input: self.spec.input_dimensions() as usize,
            output: self.spec.output_dimensions() as usize,
            groups: self.spec.group_count() as usize,
            reduction: self.spec.reduction(),
        }
    }

    fn project_slots(
        &self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
        observer: Option<&mut dyn GroupedUnitObserver<CandleTensor>>,
    ) -> Result<GroupedRows, Error> {
        self.geometry().project_slots(
            input,
            selections,
            context,
            observer,
            |group, rows| {
                let (gate, up) = match &self.packed {
                    Some(bank) => {
                        let intermediate = self.spec.intermediate_dimensions() as usize;
                        let gate_up = bank.gate_up.project(Some(group), rows, context)?;
                        let component = |offset| {
                            CandleTensor::wrap(gate_up.value.narrow(1, offset, intermediate))
                        };
                        (component(0)?, component(intermediate)?)
                    }
                    None => {
                        let expert = &self.independent[group];
                        (
                            expert.gate.project(None, rows, context)?,
                            expert.up.project(None, rows, context)?,
                        )
                    }
                };
                operators::gated_product(&gate, &up, self.spec.policy())
            },
            |group, activated| match &self.packed {
                Some(bank) => bank.down.project(Some(group), &activated, context),
                None => self.independent[group]
                    .down
                    .project(None, &activated, context),
            },
        )
    }
}

impl GroupedGatedProductOperator<CandleTensor> for CandleGatedProductGroups {
    fn spec(&self) -> &GroupedGatedProductSpec {
        &self.spec
    }

    fn forward_grouped(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        self.forward_grouped_with_unit_observer(input, selections, context, None)
    }

    fn forward_grouped_with_unit_observer(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
        observer: Option<&mut dyn GroupedUnitObserver<CandleTensor>>,
    ) -> Result<CandleTensor, Error> {
        let rows = self.project_slots(input, selections, context, observer)?;
        let mut shape = input.value.dims().to_vec();
        *shape.last_mut().ok_or_else(unavailable)? = self.spec.output_dimensions() as usize;
        self.geometry()
            .reduce_rows(rows, selections, &shape, input.value.dtype(), context)
    }

    fn forward_uncombined(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
        observer: Option<&mut dyn GroupedUnitObserver<CandleTensor>>,
    ) -> Result<CandleTensor, Error> {
        let rows = self.project_slots(input, selections, context, observer)?;
        CandleTensor::wrap(
            NativeTensor::from_vec(
                rows.outputs,
                (
                    rows.tokens,
                    rows.slots,
                    self.spec.output_dimensions() as usize,
                ),
                &context.device,
            )
            .and_then(|value| value.to_dtype(input.value.dtype())),
        )
    }

    fn combine_outputs(
        &self,
        outputs: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let (tokens, slots, width) = outputs.value.dims3().map_err(Error::backend_source)?;
        let ids = selections.group_indices().to_i32_vec(context)?;
        if slots == 0
            || width != self.spec.output_dimensions() as usize
            || selections.group_indices().shape() != selections.coefficients().shape()
            || selections.group_indices().value.dims().last() != Some(&slots)
            || ids.len() != tokens * slots
        {
            return Err(Error::backend("invalid uncombined grouped output geometry"));
        }
        let mut shape = selections.group_indices().value.dims().to_vec();
        *shape.last_mut().ok_or_else(unavailable)? = width;
        self.geometry().reduce_rows(
            GroupedRows {
                tokens,
                slots,
                ids,
                outputs: outputs.to_f32_vec(context)?,
            },
            selections,
            &shape,
            outputs.value.dtype(),
            context,
        )
    }
}

/// Activated selected projections in a packed group bank.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleLinearGroups {
    projection: GroupedProjection,
    #[parameter(skip)]
    spec: GroupedLinearSpec,
}

impl GroupedLinearOperator<CandleTensor> for CandleLinearGroups {
    fn spec(&self) -> &GroupedLinearSpec {
        &self.spec
    }

    fn forward_grouped(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let (tokens, _) = input.value.dims2().map_err(Error::backend_source)?;
        let geometry = GroupGeometry {
            input: self.spec.input_dimensions() as usize,
            output: self.spec.output_dimensions() as usize,
            groups: self.spec.group_count() as usize,
            reduction: self.spec.reduction(),
        };
        let rows = geometry.project_slots(
            input,
            selections,
            context,
            None,
            |group, rows| {
                let projected = self.projection.project(Some(group), rows, context)?;
                if self.spec.activation() == eredu_nn::GroupedLinearActivation::Silu {
                    <CandleNeuralBackend as eredu_nn::NeuralBackend>::silu(projected, context)
                } else {
                    Ok(projected)
                }
            },
            |_, projected| Ok(projected),
        )?;
        geometry.reduce_rows(
            rows,
            selections,
            &[tokens, geometry.output],
            input.value.dtype(),
            context,
        )
    }
}

/// Packed group bank with squared ReLU between its projections.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleRelu2Groups {
    up: GroupedProjection,
    down: GroupedProjection,
    #[parameter(skip)]
    spec: GroupedRelu2Spec,
}

impl GroupedRelu2Operator<CandleTensor> for CandleRelu2Groups {
    fn spec(&self) -> &GroupedRelu2Spec {
        &self.spec
    }

    fn forward_grouped(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        self.forward_grouped_with_unit_observer(input, selections, context, None)
    }

    fn forward_grouped_with_unit_observer(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        context: &CandleContext,
        observer: Option<&mut dyn GroupedUnitObserver<CandleTensor>>,
    ) -> Result<CandleTensor, Error> {
        let hidden = self.spec.hidden_dimensions() as usize;
        let geometry = GroupGeometry {
            input: hidden,
            output: hidden,
            groups: self.spec.group_count() as usize,
            reduction: GroupReduction::Sum,
        };
        let rows = geometry.project_slots(
            input,
            selections,
            context,
            observer,
            |group, rows| {
                let up = self.up.project(Some(group), rows, context)?;
                CandleTensor::wrap(up.value.relu().and_then(|value| value.sqr()))
            },
            |group, activated| self.down.project(Some(group), &activated, context),
        )?;
        geometry.reduce_rows(
            rows,
            selections,
            input.value.dims(),
            input.value.dtype(),
            context,
        )
    }
}

impl GroupedNeuralBackend for CandleNeuralBackend {
    type LinearGroups = CandleLinearGroups;
    type Selector = CandleSelector;
    type GatedProductGroups = CandleGatedProductGroups;
    type Relu2Groups = CandleRelu2Groups;

    fn grouped_linear_bank(
        spec: GroupedLinearSpec,
        context: &CandleContext,
    ) -> Result<Self::LinearGroups, Error> {
        spec.validate()?;
        let projection = GroupedProjection::new(
            spec.projection(),
            Some(spec.group_count()),
            spec.output_dimensions(),
            spec.input_dimensions(),
            context,
        )?;
        Ok(CandleLinearGroups { spec, projection })
    }

    fn grouped_linear(
        linear: &mut CandleLinear,
        input: &CandleTensor,
        groups: i32,
        output_per_group: i32,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        Self::grouped_linear_with_input_observer(
            linear,
            input,
            groups,
            output_per_group,
            context,
            None,
        )
    }

    fn grouped_linear_with_input_observer(
        linear: &mut CandleLinear,
        input: &CandleTensor,
        groups: i32,
        output_per_group: i32,
        context: &CandleContext,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<CandleTensor>>,
    ) -> Result<CandleTensor, Error> {
        eredu_nn::project_group_diagonal(linear, input, groups, output_per_group, context, observer)
    }
    fn top_k_group_selector(
        spec: TopKGroupSelectorSpec,
        context: &CandleContext,
    ) -> Result<Self::Selector, Error> {
        spec.validate()?;
        let projection = CandleLinear::new(
            LinearSpec {
                input: spec.input_dimensions(),
                output: spec.selection().group_count(),
                weight: spec.weight().clone(),
                bias: spec.bias().cloned(),
                format: spec.format().clone(),
            },
            context,
        )?;
        let slot = |parameter: &eredu_nn::ParameterSpec, width| {
            CandleTensor::unloaded_f32(&[width], context)
                .map(|value| Parameter::new(parameter.clone(), value))
        };
        let groups = spec.selection().group_count();
        let correction = spec
            .correction_bias()
            .map(|p| slot(p, groups))
            .transpose()?;
        let input_scale = spec
            .input_transform()
            .map(|t| slot(t.scale(), spec.input_dimensions()))
            .transpose()?;
        let coefficient_scale = spec
            .coefficient_scale()
            .map(|p| slot(p, groups))
            .transpose()?;
        Ok(CandleSelector {
            spec,
            projection,
            correction,
            input_scale,
            coefficient_scale,
        })
    }
    fn grouped_gated_product(
        spec: GroupedGatedProductSpec,
        context: &CandleContext,
    ) -> Result<Self::GatedProductGroups, Error> {
        spec.validate()?;
        let width = spec.input_dimensions();
        let hidden = spec.intermediate_dimensions();
        let output = spec.output_dimensions();
        let groups = Some(spec.group_count());
        let (packed, independent) = match spec.layout() {
            GatedProductGroupLayout::Packed { gate_up, down } => (
                Some(PackedGatedProduct {
                    gate_up: GroupedProjection::new(
                        gate_up,
                        groups,
                        hidden.checked_mul(2).ok_or_else(unavailable)?,
                        width,
                        context,
                    )?,
                    down: GroupedProjection::new(down, groups, output, hidden, context)?,
                }),
                Vec::new(),
            ),
            GatedProductGroupLayout::Independent(experts) => (
                None,
                experts
                    .iter()
                    .map(|expert| {
                        Ok(IndependentGatedProduct {
                            gate: GroupedProjection::new(
                                expert.gate(),
                                None,
                                hidden,
                                width,
                                context,
                            )?,
                            up: GroupedProjection::new(expert.up(), None, hidden, width, context)?,
                            down: GroupedProjection::new(
                                expert.down(),
                                None,
                                output,
                                hidden,
                                context,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?,
            ),
            _ => return Err(unavailable()),
        };
        Ok(CandleGatedProductGroups {
            spec,
            packed,
            independent,
        })
    }

    fn grouped_relu2(
        spec: GroupedRelu2Spec,
        context: &CandleContext,
    ) -> Result<Self::Relu2Groups, Error> {
        spec.validate()?;
        let (groups, hidden, intermediate) = (
            spec.group_count(),
            spec.hidden_dimensions(),
            spec.intermediate_dimensions(),
        );
        let up = GroupedProjection::new(spec.up(), Some(groups), intermediate, hidden, context)?;
        let down =
            GroupedProjection::new(spec.down(), Some(groups), hidden, intermediate, context)?;
        Ok(CandleRelu2Groups { spec, up, down })
    }
    fn joint_group_selection(
        input: JointGroupSelectionInput<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<JointGroupSelection<CandleTensor>, Error> {
        input.validate()?;
        let width = input
            .hidden()
            .value
            .dim(candle_core::D::Minus1)
            .map_err(Error::backend_source)?;
        let tokens = input.hidden().value.elem_count() / width;
        let hidden = CandleTensor::wrap(input.hidden().value.reshape((tokens, width)))?;
        let logits = CandleTensor::linear(&hidden, input.weight(), None, context)?;
        let dtype = logits.value.dtype();
        let selectable = input.selectable_groups() as usize;
        let always_on = input.always_on_groups() as usize;
        let slots = input.top_k() as usize;
        let bias = input.correction_bias().to_f32_vec(context)?;
        let scale = input.global_scale().to_f32_vec(context)?[0];
        let mut ids = Vec::with_capacity(tokens * slots);
        let mut selected = Vec::with_capacity(tokens * (slots + always_on));
        for row in logits
            .to_f32_vec(context)?
            .chunks_exact(selectable + always_on)
        {
            let ranking: Vec<_> = row[..selectable]
                .iter()
                .zip(&bias)
                .map(|(&logit, &bias)| {
                    round_float(eredu_nn::routing_selection::sigmoid(logit), dtype) + bias
                })
                .collect();
            if ranking.iter().any(|score| !score.is_finite())
                || row.iter().any(|score| !score.is_finite())
            {
                return Err(Error::backend("joint routing scores must be finite"));
            }
            let mut ranked: Vec<_> = (0..selectable).collect();
            let start = selectable - slots;
            eredu_nn::routing_selection::partition_ids(&ranking, &mut ranked, start);
            for &id in &ranked[start..] {
                ids.push(id as i32);
            }
            selected.extend(
                ranked[start..]
                    .iter()
                    .map(|&id| row[id])
                    .chain(row[selectable..].iter().copied())
                    .map(|logit| -eredu_nn::routing_selection::softplus(-logit)),
            );
        }
        let logits = CandleTensor::wrap(
            NativeTensor::from_vec(selected, (tokens, slots + always_on), &context.device)
                .and_then(|value| value.to_dtype(dtype)),
        )?;
        let coefficients = CandleSelector::softmax(&logits)?;
        let first_dtype = dtype;
        let final_dtype = input.global_scale().value.dtype();
        let values = coefficients
            .to_f32_vec(context)?
            .into_iter()
            .map(|value| {
                round_float(
                    value * round_float(input.coefficient_scale(), first_dtype),
                    first_dtype,
                ) * scale
            })
            .collect::<Vec<_>>();
        let ids = CandleTensor::wrap(NativeTensor::from_vec(
            ids,
            (tokens, slots),
            &context.device,
        ))?;
        let coefficients =
            NativeTensor::from_vec(values, (tokens, slots + always_on), &context.device)
                .and_then(|value| {
                    value.to_dtype(if final_dtype == DType::F32 {
                        final_dtype
                    } else {
                        dtype
                    })
                })
                .map_err(Error::backend_source)?;
        Ok(JointGroupSelection::new(
            ids,
            CandleTensor::wrap(coefficients.narrow(1, 0, slots))?,
            CandleTensor::wrap(coefficients.narrow(1, slots, always_on))?,
        ))
    }
}

impl eredu_nn::TensorParallelGroupedGatedProductOperator<CandleTensor>
    for CandleGatedProductGroups
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        partitions: usize,
        context: &CandleContext,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<CandleTensor>, Error> {
        if partitions != 1 {
            return Err(unavailable());
        }
        self.forward_grouped(input, selections, context)
            .map(|output| eredu_nn::TensorParallelGroupedOutput::new(output, None))
    }
}

impl eredu_nn::TensorParallelGroupedRelu2Operator<CandleTensor> for CandleRelu2Groups {
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &CandleTensor,
        selections: &GroupSelection<CandleTensor>,
        partitions: usize,
        context: &CandleContext,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<CandleTensor>, Error> {
        if partitions != 1 {
            return Err(unavailable());
        }
        self.forward_grouped(input, selections, context)
            .map(|output| eredu_nn::TensorParallelGroupedOutput::new(output, None))
    }
}

fn compact_name(id: &str) -> &str {
    id.rsplit('.')
        .next()
        .expect("parameter IDs have at least one component")
}

fn compact_names(bank: &impl Parameterized<CandleTensor>) -> Vec<String> {
    struct Slots(Vec<String>);
    impl eredu_nn::ParameterVisitor<'_, CandleTensor> for Slots {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadata, _: &CandleTensor) {
            self.0.push(compact_name(metadata.id.as_str()).to_owned());
        }
    }
    let mut names = Slots(Vec::new());
    bank.visit_parameters(&mut names);
    names.0
}

fn bind_compact(
    bank: &mut (impl Parameterized<CandleTensor> + Clone),
    values: std::collections::BTreeMap<String, CandleTensor>,
) -> Result<(), Error> {
    let names = compact_names(bank);
    if names.len() != values.len()
        || names
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != names.len()
    {
        return Err(unavailable());
    }
    struct Install<'a>(&'a std::collections::BTreeMap<String, CandleTensor>, bool);
    impl eredu_nn::ParameterVisitorMut<'_, CandleTensor> for Install<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadata, slot: &mut CandleTensor) {
            if let Some(value) = self
                .0
                .get(compact_name(metadata.id.as_str()))
                .filter(|value| value.shape() == slot.shape())
            {
                slot.publish_parameter(value);
            } else {
                self.1 = false;
            }
        }
    }
    let mut candidate = bank.clone();
    let mut install = Install(&values, true);
    candidate.visit_parameters_mut(&mut install);
    if !install.1 {
        return Err(unavailable());
    }
    *bank = candidate;
    Ok(())
}

macro_rules! compact_group_bindings {
    ($($bank:ty),+ $(,)?) => {$(
        impl eredu_core::residency::CompactGroupBindings<CandleTensor> for $bank {
            fn binding_names(&self) -> Vec<String> {
                compact_names(self)
            }
            fn bind_compact(
                &mut self,
                values: std::collections::BTreeMap<String, CandleTensor>,
            ) -> Result<(), Error> {
                bind_compact(self, values)
            }
        }
    )+};
}

compact_group_bindings!(
    CandleGatedProductGroups,
    CandleLinearGroups,
    CandleRelu2Groups
);
