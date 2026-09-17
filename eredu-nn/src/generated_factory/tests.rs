use super::*;
use crate::{reconstruct_block_fp8_input_retained, BlockFp8InputReconstructionMechanism};
use std::{cell::Cell, rc::Rc};

// Value deliberately has no Clone implementation: retention and moved return
// ownership are distinct operations.
struct Value(Rc<usize>);
struct Program<'a> {
    created: &'a Cell<usize>,
}
impl Program<'_> {
    fn next(&self) -> Value {
        let n = self.created.get() + 1;
        self.created.set(n);
        Value(Rc::new(n))
    }
}
impl BlockFp8InputReconstructionMechanism for Program<'_> {
    type Value = Value;
    type Error = u32;
    fn expand_scales(&self) -> Result<Value, u32> {
        Ok(self.next())
    }
    fn broadcast_scales(&self, _: &Value) -> Result<Value, u32> {
        Ok(self.next())
    }
    fn flatten_scales(&self, _: &Value) -> Result<Value, u32> {
        Ok(self.next())
    }
    fn decode_values(&self) -> Result<Value, u32> {
        Ok(self.next())
    }
    fn trim_scales(&self, _: &Value) -> Result<Value, u32> {
        Ok(self.next())
    }
    fn multiply(&self, _: &Value, _: &Value) -> Result<Value, u32> {
        Ok(self.next())
    }
    fn restore_shape(&self, _: &Value) -> Result<Value, u32> {
        Ok(self.next())
    }
}
#[test]
fn retention_precedes_each_next_operation_and_partial_roots_outlive_error() {
    for stop in 1..=8 {
        let created = Cell::new(0);
        let mut retained = Vec::new();
        let result =
            reconstruct_block_fp8_input_retained(Program { created: &created }, &mut |value| {
                assert_eq!(created.get(), retained.len() + 1);
                retained.push(value.0.clone());
                if retained.len() == stop {
                    Err(73)
                } else {
                    Ok(())
                }
            });
        assert_eq!(created.get(), stop.min(7));
        assert_eq!(retained.len(), stop.min(7));
        if stop <= 7 {
            assert!(matches!(result, Err(73)));
        } else {
            let output = result.unwrap_or_else(|_| panic!("complete"));
            assert!(Rc::ptr_eq(&output.0, retained.last().unwrap()));
            drop(output);
        }
        let weak = retained.iter().map(Rc::downgrade).collect::<Vec<_>>();
        assert!(retained.iter().all(|value| Rc::strong_count(value) == 1));
        drop(retained);
        assert!(weak.iter().all(|value| value.upgrade().is_none()));
    }
}

struct Factory<'a> {
    shape: &'a [i32],
    inputs: [Value; 2],
    calls: usize,
    swallow: bool,
}
impl RetainedGeneratedTensorFactory<Value, u32> for Factory<'_> {
    fn program(&self) -> GeneratedTensorProgram<'_> {
        GeneratedTensorProgram::BlockFp8Input(
            BlockFp8InputReconstructionPlan::new(self.shape).unwrap(),
        )
    }
    fn visit_sources(
        &mut self,
        retain: &mut dyn FnMut(GeneratedTensorSourceRole, &Value) -> Result<(), u32>,
    ) -> Result<(), u32> {
        retain(GeneratedTensorSourceRole::CompactValues, &self.inputs[0])?;
        retain(GeneratedTensorSourceRole::BlockScales, &self.inputs[1])
    }
    fn generate(
        &mut self,
        retain: &mut dyn FnMut(&Value) -> Result<(), u32>,
    ) -> Result<Value, u32> {
        self.calls += 1;
        let value = Value(Rc::new(9));
        let result = retain(&value);
        if !self.swallow {
            result?;
        }
        Ok(value)
    }
}
#[derive(Debug)]
struct Original(Rc<u32>);
fn value_ref(value: &Value) -> &Value {
    value
}
#[test]
fn mapped_factory_preserves_source_identity_and_original_sink_error() {
    let mut factory = Factory {
        shape: &[2, 259],
        inputs: [Value(Rc::new(1)), Value(Rc::new(2))],
        calls: 0,
        swallow: true,
    };
    let first = factory.inputs[0].0.clone();
    let original = Rc::new(83);
    {
        let mut mapped = MappedGeneratedTensorFactory::new(
            &mut factory,
            value_ref,
            std::convert::identity::<Value>,
            |error| Original(Rc::new(error)),
            |_: &Original| 91,
        );
        let GeneratedTensorProgram::BlockFp8Input(plan) = mapped.program();
        assert_eq!(plan.width(), 259);
        let error = mapped
            .visit_sources(&mut |role, value| {
                assert_eq!(role, GeneratedTensorSourceRole::CompactValues);
                assert!(Rc::ptr_eq(&first, &value.0));
                Err(Original(original.clone()))
            })
            .unwrap_err();
        assert!(Rc::ptr_eq(&error.0, &original));
        let error = match mapped.generate(&mut |_| Err(Original(original.clone()))) {
            Err(error) => error,
            Ok(_) => panic!("swallowed propagation signal must still fail"),
        };
        assert!(Rc::ptr_eq(&error.0, &original));
    }
    assert_eq!(factory.calls, 1);
}
