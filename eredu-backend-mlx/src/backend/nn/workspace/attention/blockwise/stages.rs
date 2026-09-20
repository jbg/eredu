//! Per-page actual buffer facts over the existing native recurrence costs.
use super::*;
use descriptor::{Descriptor, Stage};
use eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry;

pub(super) fn mask_bytes(
    g: AbsoluteAttentionMaskGeometry,
    a: NativeAllocationFacts,
) -> FactResult<u64> {
    let queries = capacity(a, g.queries() as u64)?;
    let keys = capacity(a, g.keys() as u64)?;
    let mask = facts::buffer_capacity(a, mul(g.queries() as u64, g.keys() as u64)?)?;
    let mut bytes = add(add(queries, keys)?, mask)?;
    if g.window().is_some() {
        // Earliest-query vector/scalar, visibility comparison, final AND.
        bytes = add(bytes, add(add(queries, capacity(a, 1)?)?, mul(2, mask)?)?)?;
        if g.prefix() > 0 {
            // Prefix column/scalar and combined visibility page.
            bytes = add(
                bytes,
                add(
                    add(facts::buffer_capacity(a, g.keys() as u64)?, capacity(a, 1)?)?,
                    mask,
                )?,
            )?;
        }
    }
    Ok(bytes)
}
pub(in crate::backend::nn::workspace::attention) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(Descriptor { policy, stage }) = descriptor::decode(operation)? else {
        return Ok(None);
    };
    let scratch = match stage {
        Stage::Begin {
            q,
            mask,
            sink: learned,
        } => {
            sink.output(Output::AllocateOrAliasInputs {
                bytes: capacity(
                    allocation,
                    mul(
                        mul(mul(q[0] as u64, q[1] as u64)?, q[2] as u64)?,
                        q[3] as u64,
                    )?,
                )?,
                inputs: Aliases::Slice(&[0]),
            })?;
            if mask.is_some() {
                sink.output(Output::AliasInput(1))?;
            }
            let _ = learned;
            0
        }
        Stage::Accumulate {
            q,
            k,
            v,
            mask,
            sink: learned,
            bias,
            previous: _,
            value_pass,
            absolute,
        } => {
            let g = Geometry {
                b: q[0],
                h: q[1],
                kv: k[1],
                q: q[2],
                k: k[2],
                d: q[3],
                v: v[3],
            };
            let mask = match mask {
                None => Mask::None,
                Some(layout) if layout.dtype() == WorkspaceDtype::Bool => Mask::Boolean,
                Some(layout) => Mask::Additive(layout.elements()?),
            };
            let (common, normalization, values) = block_costs(
                g,
                mask,
                learned.is_some(),
                policy.options.softcap.is_some(),
                allocation,
                policy.options.arithmetic == AttentionArithmetic::InputScores,
                Some(absolute),
                bias.is_some(),
            )?;
            let row = capacity(allocation, g.rows()?)?;
            let output = capacity(allocation, g.output()?)?;
            if !value_pass {
                sink.output(Output::Allocate(row))?;
                sink.output(Output::Allocate(row))?;
            }
            sink.output(Output::Allocate(output))?;
            let outputs = if value_pass {
                output
            } else {
                add(mul(2, row)?, output)?
            };
            add(common, if value_pass { values } else { normalization })?
                .checked_sub(outputs)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?
        }
        Stage::Finish {
            shape,
            input_scores,
        } => {
            let rows = mul(mul(shape[0] as u64, shape[1] as u64)?, shape[2] as u64)?;
            let output = capacity(allocation, mul(rows, shape[3] as u64)?)?;
            if input_scores {
                sink.output(Output::AllocateOrAliasInputs {
                    bytes: output,
                    inputs: Aliases::Slice(&[0]),
                })?;
                0
            } else {
                sink.output(Output::Allocate(output))?;
                // Safe denominator comparison/where, two scalars and FP32
                // division before a possible final narrower native cast.
                add(
                    add(
                        mul(3, capacity(allocation, rows)?)?,
                        mul(2, capacity(allocation, 1)?)?,
                    )?,
                    output,
                )?
            }
        }
    };
    sink.finish(scratch,format_args!("shared native paged blockwise recurrence: actual page geometry, integer causal/window/prefix mask, grouped-head source views, floating casts, score rounding/cap/bias, normalization and weighted-value buffers; each page completes before next page allocation; four-byte floating population covers narrower native scalars")).map(Some)
}
