use super::*;

pub(super) struct Projection<'a> {
    format: &'a LinearFormatSpec,
    groups: u64,
    columns: u64,
    rows: u64,
    weight: WorkspaceLayoutView<'a>,
    companions: WorkspaceLayoutList<'a>,
    pub(super) bias: Option<WorkspaceLayoutView<'a>>,
}
impl<'a> Projection<'a> {
    pub(super) fn new(
        spec: &'a GroupedProjectionSpec,
        groups: i32,
        columns: i32,
        rows: i32,
        inputs: WorkspaceLayoutList<'a>,
        slot: &mut usize,
        a: NativeAllocationFacts,
    ) -> FactResult<Option<Self>> {
        let format = spec.format();
        let count = 1
            + usize::from(format.scale().is_some())
            + usize::from(format.affine_bias().is_some())
            + usize::from(spec.bias().is_some());
        let end = slot.checked_add(count).ok_or_else(invalid)?;
        let parameters = inputs.slice(*slot..end).ok_or_else(invalid)?;
        let input_shape = [1, columns];
        let input = WorkspaceLayoutView::new(&input_shape, WorkspaceDtype::Float32)?;
        // A projection has one input and at most four actual parameter roles.
        // Parameter tails borrow their complete rank from the retained source.
        let mut ungrouped = [input; 5];
        for (index, value) in parameters.iter().enumerate() {
            if value.shape().first() != Some(&groups) {
                return Err(invalid());
            }
            ungrouped[index + 1] = WorkspaceLayoutView::new(&value.shape()[1..], value.dtype())?;
        }
        let output_shape = [1, rows];
        let outputs = [WorkspaceLayoutView::new(
            &output_shape,
            WorkspaceDtype::Float32,
        )?];
        let projection = WorkspaceOperationView {
            kind: WorkspaceOperationKindView::Projection(format),
            inputs: WorkspaceLayoutList::Views(&ungrouped[..count + 1]),
            outputs: WorkspaceLayoutList::Views(&outputs),
        };
        if let LinearFormat::E4M3BlockFp8(config) = format.encoding() {
            // Grouped FP8 additionally retains independent row-block origins.
            if config.block_rows != 128 || config.block_columns != 128 {
                return Ok(None);
            }
            let scale_rows = format.row_layout().scale_rows_fixed(rows as usize, 128)?;
            let scale_dtype = if config.scale_encoding == BlockFp8ScaleEncoding::Ue8m0 {
                WorkspaceDtype::Uint8
            } else {
                WorkspaceDtype::Float32
            };
            if parameters.get(0).unwrap().shape() != [groups, rows, columns]
                || parameters.get(0).unwrap().dtype() != WorkspaceDtype::Uint8
                || parameters.get(1).unwrap().shape()
                    != [
                        groups,
                        i32::try_from(scale_rows).map_err(|_| invalid())?,
                        (columns as u32).div_ceil(128) as i32,
                    ]
                || parameters.get(1).unwrap().dtype() != scale_dtype
            {
                return Err(invalid());
            }
            if spec.bias().is_some()
                && (parameters.last().unwrap().shape() != [groups, rows]
                    || parameters.last().unwrap().dtype() != WorkspaceDtype::Float32)
            {
                return Err(invalid());
            }
        } else {
            let bound = if format.encoding() == LinearFormat::Dense {
                matrix::emit(projection, a, &mut Emitter::count())?
            } else {
                packed::emit(projection, a, &mut Emitter::count())?
            };
            if bound.is_none() {
                return Ok(None);
            }
        }
        *slot = end;
        let companion_end = count - usize::from(spec.bias().is_some());
        Ok(Some(Self {
            format,
            groups: groups as u64,
            columns: columns as u64,
            rows: rows as u64,
            weight: parameters.get(0).unwrap(),
            companions: parameters.slice(1..companion_end).unwrap(),
            bias: spec.bias().map(|_| parameters.get(count - 1).unwrap()),
        }))
    }
    // Same dense branch selected by packed_grouped_linear when a reversible
    // overlay publishes floating weights. Its logical bank is independent of
    // the smaller retained packed U32 storage and its U8 companions.
    fn dense_cost(
        &self,
        n: u64,
        a: NativeAllocationFacts,
        custom: bool,
    ) -> FactResult<Option<Cost>> {
        let mut cost = Cost::new(a);
        let input = mul(n, self.columns)?;
        let output = mul(n, self.rows)?;

        // GatherMM promotes each original operand before backend
        // compaction. It reads the bank directly and has no split-K
        // accumulator or selection-sized dense weight expansion.
        cost.buffers(input, 3)?; // reshape, promotion, compaction
        cost.buffers(mul(self.groups, mul(self.rows, self.columns)?)?, 2)?;
        cost.buffers(n, 4)?; // default lhs IDs, casts and compact rhs IDs
        cost.buffers(output, 2)?; // kernel and result reshape
        cost.buffers(1, 2)?;
        if custom {
            if !crate::backend::nn::matrix::bf16_row_width_supported(i32::try_from(self.columns)?)
                || n == 0
            {
                return Ok(None);
            }
            let mut custom = Cost::new(a);
            custom.buffers(input, 1)?;
            custom.buffers(mul(self.groups, mul(self.rows, self.columns)?)?, 1)?;
            custom.buffers(n, 2)?;
            custom.buffers(output, 2)?;
            // The BF16 row kernel validates native group IDs before
            // dispatch. Include every predicate and its reduction; original
            // construction also retains safe masked IDs until completion.
            custom.pointwise(n, 1)?;
            custom.pointwise(n, 1)?;
            custom.pointwise(n, 2)?;
            custom.sum(n, 1, n)?;
            // Original row projection retains an invalid predicate and
            // masks unsafe IDs before dispatch: logical_not, zeros_like
            // and where. The existing comparison/sum terms stay live.
            custom.buffers(n, 3)?;
            custom.buffers(1, 1)?;
            custom.default_scalars(if a.original_storage { 3 } else { 2 })?;
            cost = custom;
        }

        Ok(Some(cost))
    }
    pub(super) fn cost(
        &self,
        n: u64,
        a: NativeAllocationFacts,
        source: usize,
    ) -> FactResult<Option<Cost>> {
        let mut cost = Cost::new(a);
        let input = mul(n, self.columns)?;
        let output = mul(n, self.rows)?;
        let encoding = self.format.encoding();
        let dense = match encoding {
            LinearFormat::Dense if source < 2 => Some(source == 1),
            LinearFormat::Affine(_) | LinearFormat::MxFp4 if source > 0 => Some(source == 2),
            _ if source == 0 => None,
            _ => return Ok(None),
        };
        if let Some(custom) = dense {
            let Some(selected) = self.dense_cost(n, a, custom)? else {
                return Ok(None);
            };
            cost = selected;
        } else {
            match encoding {
                LinearFormat::Dense => unreachable!(),
                LinearFormat::Affine(config) if config.group_size == 16 => {
                    // This selected mechanism really gathers every packed expert
                    // matrix and companion before batched qmv. Count those replicas
                    // and subsequent promotions/compaction independently.
                    cost.buffers(input, 3)?;
                    let weight_row = self
                        .weight
                        .bytes()?
                        .checked_div(self.groups)
                        .ok_or_else(invalid)?;
                    cost.bytes(mul(n, weight_row)?, 2)?;
                    for companion in self.companions.iter() {
                        let row = companion
                            .bytes()?
                            .checked_div(self.groups)
                            .ok_or_else(invalid)?;
                        cost.bytes(mul(n, row)?, 3)?;
                    }
                    cost.buffers(n, 2)?;
                    cost.buffers(output, 2)?;
                    cost.buffers(1, 2)?;
                }
                LinearFormat::Affine(_) | LinearFormat::MxFp4 => {
                    cost.buffers(input, 3)?;
                    cost.bytes(self.weight.bytes()?, 1)?;
                    for companion in self.companions.iter() {
                        cost.bytes(
                            companion.bytes()?,
                            if companion.dtype() == WorkspaceDtype::Float32 {
                                2
                            } else {
                                1
                            },
                        )?;
                    }
                    cost.buffers(n, 4)?; // explicit lhs arange, index casts and possible compaction
                    cost.buffers(output, 2)?;
                    cost.buffers(1, 2)?;
                }
                LinearFormat::GgufIQuant { .. } => {
                    // All selected GGML kernels decode packed data in registers.
                    // Custom-kernel preparation may compact each input once.
                    cost.buffers(input, 1)?;
                    cost.bytes(self.weight.bytes()?, 1)?;
                    cost.buffers(n, 1)?;
                    cost.buffers(output, 1)?;
                }
                LinearFormat::E4M3BlockFp8(config) => {
                    cost.buffers(input, 1)?; // activation quantizer input compaction
                    cost.bytes(input, 2)?; // quantized activation and possible projector copy
                    cost.buffers(mul(n, self.columns.div_ceil(128))?, 2)?;
                    cost.bytes(self.weight.bytes()?, 2)?; // partition-shape copy and projector compaction
                    let scales = self.companions.get(0).unwrap().elements()?;
                    cost.buffers(scales, 2)?; // partition-shape copy and projector compaction
                    if config.scale_encoding == BlockFp8ScaleEncoding::Ue8m0 {
                        // The shared scale decoder borrows its process-owned table.
                        cost.buffers(256, 1)?;
                        cost.buffers(scales, 2)?; // indices and decoded scale values
                    }
                    cost.buffers(n, 1)?; // group-ID custom-kernel compaction
                    cost.buffers(output, 2)?; // F32 result and activation dtype restoration
                    cost.buffers(1, 2)?;
                }
            }
        }
        if self.bias.is_some() {
            cost.buffers(output, 1)?; // direct bias gather
            cost.pointwise(output, 2)?;
        }
        Ok(Some(cost))
    }
}
