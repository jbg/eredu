//! Safe callers of the shared paged online-softmax worker.
use super::*;
use crate::backend::nn::workspace::attention::blockwise::descriptor::{self, Descriptor, Stage};
use OrdinaryRecipeCall as C;

#[derive(Default)]
struct Calls(OrdinaryCallControls);
impl Calls {
    fn add(&mut self, call: C) -> Option<()> {
        self.0 = self.0.append(OrdinaryCallControls::call(call)?)?;
        Some(())
    }
    fn repeated(&mut self, call: C, count: usize) -> Option<()> {
        self.0 = self
            .0
            .append(OrdinaryCallControls::call(call)?.repeat(count)?)?;
        Some(())
    }
    fn scalar_binary(&mut self) -> Option<()> {
        self.add(C::ScalarF32)?;
        self.add(C::Binary)
    }
    fn aliases(&mut self, count: usize) -> Option<()> {
        self.0 = self
            .0
            .metadata(Array::ordinary_clone_control_bytes()?.checked_mul(count)?)?;
        Some(())
    }
    fn safe_denominator(&mut self) -> Option<()> {
        self.scalar_binary()?;
        self.add(C::ScalarF32)?;
        self.add(C::Select)
    }
}

pub(super) fn calls(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let Descriptor { policy, stage } = descriptor::decode(operation).ok()??;
    let mut calls = Calls::default();
    match stage {
        Stage::Begin { mask, sink, .. } => {
            calls.add(C::Cast)?;
            if mask.is_some() {
                calls.add(C::Broadcast { rank: 4 })?;
            }
            calls.aliases(usize::from(sink.is_some()))?;
        }
        Stage::Finish { input_scores, .. } => {
            if !input_scores {
                calls.safe_denominator()?;
                calls.add(C::Binary)?;
            }
            calls.add(C::Cast)?;
        }
        Stage::Accumulate {
            q,
            k,
            mask,
            sink,
            bias,
            previous,
            value_pass,
            absolute,
            ..
        } => {
            // The actual pair either clones its two handles or expands each
            // KV head through reshape, broadcast, then the rank-four reshape.
            if q[1] == k[1] {
                calls.aliases(2)?;
            } else {
                calls.repeated(C::Reshape { rank: 5 }, 2)?;
                calls.repeated(C::Broadcast { rank: 5 }, 2)?;
                calls.repeated(C::Reshape { rank: 4 }, 2)?;
            }
            calls.repeated(C::Cast, 2)?;
            calls.add(C::SwapAxes)?;
            calls.add(C::Binary)?; // QK product
            calls.scalar_binary()?; // query or score scaling
            let input_scores =
                policy.options.arithmetic == eredu_nn::AttentionArithmetic::InputScores;
            if input_scores {
                calls.repeated(C::Cast, 3)?;
            }
            if policy.options.softcap.is_some() {
                calls.scalar_binary()?;
                calls.add(C::Unary)?;
                calls.scalar_binary()?;
            }
            if bias.is_some() {
                calls.add(C::Cast)?;
                calls.add(C::Binary)?;
            }
            // The descriptor has authenticated the integer-coordinate mask
            // producer, including its finite window and prefix branches.
            calls.repeated(C::ArangeI32, 2)?;
            calls.repeated(C::Reshape { rank: 2 }, 2)?;
            calls.add(C::Binary)?;
            if absolute.window().is_some() {
                calls.add(C::ScalarI32)?;
                calls.repeated(C::Binary, 3)?;
                if absolute.prefix() > 0 {
                    calls.add(C::ScalarI32)?;
                    calls.repeated(C::Binary, 2)?;
                }
            }
            if let Some(mask) = mask {
                calls.add(C::StaticSlice { rank: 4 })?;
                calls.add(C::Binary)?; // allowed AND explicit mask
                if mask.dtype() != WorkspaceDtype::Bool {
                    calls.repeated(C::Unary, 2)?; // is_neg_inf, logical_not
                    calls.add(C::Cast)?;
                    calls.add(C::Binary)?; // additive mask
                }
            }
            calls.add(C::ScalarF32)?;
            calls.add(C::Select)?;
            calls.add(C::Cast)?;
            if value_pass {
                calls.safe_denominator()?;
                calls.add(C::Binary)?; // score minus global maximum
                calls.add(C::Unary)?;
                calls.add(C::Cast)?; // visibility
                calls.repeated(C::Binary, 2)?; // mask and denominator
                calls.repeated(C::Cast, 2)?; // probability rounding, FP32 product
                calls.add(C::Binary)?;
                if previous {
                    calls.add(C::Binary)?;
                }
            } else {
                calls.add(C::ReduceAxis)?;
                calls.add(C::Binary)?;
                calls.add(C::Unary)?;
                calls.add(C::Cast)?;
                calls.add(C::Binary)?;
                calls.add(C::ReduceAxis)?;
                calls.add(if input_scores {
                    C::Fill { rank: 4 }
                } else {
                    C::Binary
                })?;
                if previous || sink.is_some() {
                    if !previous {
                        calls.add(C::Cast)?;
                        calls.add(C::Reshape { rank: 4 })?;
                        calls.add(C::Broadcast { rank: 4 })?;
                    }
                    calls.add(C::Binary)?; // joint maximum
                    calls.repeated(C::Binary, 2)?;
                    calls.repeated(C::Unary, 2)?;
                    calls.repeated(C::Binary, if previous { 6 } else { 3 })?;
                }
            }
            calls.0 = calls.0.append(
                crate::backend::runtime::cache::residency::ordinary_cache_evaluation_call_controls(
                    if value_pass { 1 } else { 3 },
                )?,
            )?;
            calls.0 = calls.0.metadata(
                crate::backend::runtime::cache::kv::BlockwiseAttentionAccumulator::ordinary_settlement_control_bytes()?,
            )?;
        }
    }
    let frames = [
        size_of::<Calls>(),
        size_of::<Option<OrdinaryCallControls>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<Descriptor<'_>>(),
        size_of::<Stage<'_>>(),
        size_of::<C>(),
        size_of::<Option<()>>(),
        size_of::<(&mut Calls, C, usize)>(),
    ];
    calls.0.metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
}
