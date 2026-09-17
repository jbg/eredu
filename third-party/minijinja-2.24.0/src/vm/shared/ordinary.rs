//! The ordinary storage adapter retains the existing Value/Object/Context paths.
use super::{Order, Storage};
use crate::error::{Error, ErrorKind};
use crate::output::Output;
use crate::utils::{UndefinedBehavior, write_escaped};
use crate::value::{UndefinedType, Value, ValueRepr, ops};
use crate::vm::{State, Vm, context::Stack};

pub(in crate::vm) struct Ordinary<'borrow, 'template, 'env, 'output> {
    pub vm: &'borrow Vm<'env>,
    pub state: &'borrow mut State<'template, 'env>,
    pub stack: &'borrow mut Stack,
    pub collections: &'borrow mut Vec<Vec<Value>>,
    pub loaded_tests: &'borrow mut [Option<&'env Value>],
    pub loaded_filters: &'borrow mut [Option<&'env Value>],
    pub output: &'borrow mut Output<'output>,
    pub next_loop_recursion_jump: &'borrow mut Option<(u32, bool)>,
    pub undefined_behavior: UndefinedBehavior,
    pub strict_undefined: bool,
}

impl<'env> Storage<'env> for Ordinary<'_, '_, 'env, '_> {
    type Value = Value;
    type Error = Error;

    fn pop(&mut self) -> Result<Value, Error> {
        Ok(self.stack.pop())
    }
    fn peek(&self) -> Result<&Value, Error> {
        Ok(self.stack.peek())
    }
    fn push(&mut self, value: Value) -> Result<(), Error> {
        self.stack.push(value);
        Ok(())
    }
    fn lookup(&mut self, name: &'env str) -> Result<Value, Error> {
        self.state
            .lookup(name)
            .unwrap_or(Value::UNDEFINED)
            .validate()
    }
    fn store(&mut self, name: &'env str, value: Value) -> Result<(), Error> {
        self.state.ctx.store(name, value);
        Ok(())
    }
    fn set_attr(&mut self, object: Value, name: &'env str, value: Value) -> Result<(), Error> {
        if let Some(ns) = object.downcast_object_ref::<crate::value::namespace_object::Namespace>()
        {
            ns.set_value(name, value);
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("can only assign to namespaces, not {}", object.kind()),
            ))
        }
    }
    fn build_map(&mut self, count: usize) -> Result<(), Error> {
        let mut map=crate::value::value_map_with_capacity(count);
        self.stack.reverse_top(count*2);
        for _ in 0..count {
            let key=self.stack.pop();let value=self.stack.pop();map.insert(key,value);
        }
        self.push(Value::from_object(map))
    }
    fn begin_collection(&mut self) -> Result<(), Error> {
        self.collections.push(Vec::new());
        Ok(())
    }
    fn append_collection(&mut self, value: Value) -> Result<(), Error> {
        self.collections.last_mut().expect("compiled collection").push(value);
        Ok(())
    }
    fn end_collection(&mut self) -> Result<Value, Error> {
        Ok(Value::from_object(self.collections.pop().expect("compiled collection")))
    }
    fn build_list(&mut self, count: usize) -> Result<(), Error> {
        let mut values = Vec::with_capacity(crate::utils::untrusted_size_hint(count));
        for _ in 0..count { values.push(self.stack.pop()); }
        values.reverse();
        self.push(Value::from_object(values))
    }
    fn build_kwargs(&mut self, count: usize) -> Result<(), Error> {
        let mut map = crate::value::value_map_with_capacity(count);
        self.stack.reverse_top(count * 2);
        for _ in 0..count {
            let key = self.stack.pop();
            let value = self.stack.pop();
            map.insert(key, value);
        }
        self.stack.push(crate::value::Kwargs::wrap(map));
        Ok(())
    }
    fn call_function(
        &mut self,
        _name: &'env str,
        arguments: Option<u16>,
        function: Option<Value>,
        _pc: u32,
    ) -> Result<Option<u32>, Error> {
        // Super and recursive Loop calls stay in their existing VM branch.
        let args = self.stack.get_call_args(arguments);
        let function = function.expect("ordinary VM resolved the function exactly once");
        let value = function.call(self.state, args)?;
        let count = args.len();
        self.stack.drop_top(count);
        self.stack.push(value);
        Ok(None)
    }
    fn duplicate(&mut self) -> Result<(), Error> {
        self.stack.push(self.stack.peek().clone());
        Ok(())
    }
    fn enclose(&mut self, name: &'env str) -> Result<(), Error> {
        #[cfg(feature = "macros")]
        {
            if self.state.ctx.closure().is_none() {
                let closure = std::sync::Arc::new(crate::vm::closure_object::Closure::default());
                self.state.closure_tracker.track_closure(closure.clone());
                self.state.ctx.reset_closure(Some(closure));
            }
            self.state.ctx.enclose(name);
            Ok(())
        }
        #[cfg(not(feature = "macros"))]
        {
            let _ = name;
            Err(Error::from(ErrorKind::InvalidOperation))
        }
    }
    fn closure(&mut self) -> Result<Value, Error> {
        #[cfg(feature = "macros")]
        {
            Ok(self
                .state
                .ctx
                .closure()
                .map_or(Value::UNDEFINED, |x| Value::from_dyn_object(x.clone())))
        }
        #[cfg(not(feature = "macros"))]
        {
            Err(Error::from(ErrorKind::InvalidOperation))
        }
    }
    fn build_macro(&mut self, name: &'env str, offset: u32, flags: u8) -> Result<(), Error> {
        #[cfg(feature = "macros")]
        {
            self.vm
                .build_macro(self.stack, self.state, offset, name, flags);
            Ok(())
        }
        #[cfg(not(feature = "macros"))]
        {
            let _ = (name, offset, flags);
            Err(Error::from(ErrorKind::InvalidOperation))
        }
    }
    fn return_macro(&mut self) -> Result<Option<u32>, Error> {
        Ok(None)
    }
    fn unpack(&mut self, count: usize) -> Result<(), Error> {
        self.vm.unpack_list(self.stack, count)
    }
    fn attr(&mut self, value: Value, name: &str) -> Result<Value, Error> {
        match value.get_attr_fast(name) {
            Some(value) => value.validate(),
            None => self
                .undefined_behavior
                .handle_undefined(value.is_undefined()),
        }
    }
    fn item(&mut self, value: Value, key: Value) -> Result<Value, Error> {
        match value.get_item_opt(&key) {
            Some(value) => value.validate(),
            None => self
                .undefined_behavior
                .handle_undefined(value.is_undefined()),
        }
    }
    fn slice(&mut self, value: Value, start: Value, stop: Value, step: Value) -> Result<Value, Error> {
        if value.is_undefined() && matches!(self.undefined_behavior, UndefinedBehavior::Strict) {
            return Err(Error::from(ErrorKind::UndefinedError));
        }
        ops::slice(value, start, stop, step)
    }
    fn call_method(&mut self, name: &'env str, arguments: Option<u16>) -> Result<(), Error> {
        let args = self.stack.get_call_args(arguments);
        let count = args.len();
        let value = args[0].call_method(self.state, name, &args[1..])?;
        self.stack.drop_top(count);
        self.stack.push(value);
        Ok(())
    }
    fn add(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        ops::add(&left, &right)
    }
    fn string_concat(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.undefined_behavior.assert_value_not_undefined(&left)?;
        self.undefined_behavior.assert_value_not_undefined(&right)?;
        Ok(ops::string_concat(left, &right))
    }
    fn arithmetic(
        &mut self,
        operation: crate::value::primitive::scalar::Arithmetic,
        left: Value,
        right: Value,
    ) -> Result<Value, Error> {
        use crate::value::primitive::scalar::Arithmetic;
        // Keep each ordinary operator's dynamic prebranches, conversions and
        // detailed errors in its existing value worker.
        match operation {
            Arithmetic::Add => ops::add(&left, &right),
            Arithmetic::Sub => ops::sub(&left, &right),
            Arithmetic::Mul => ops::mul(&left, &right),
            Arithmetic::Div => ops::div(&left, &right),
            Arithmetic::FloorDiv => ops::int_div(&left, &right),
            Arithmetic::Rem => ops::rem(&left, &right),
            Arithmetic::Pow => ops::pow(&left, &right),
        }
    }
    fn ne(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.undefined_behavior.assert_value_not_undefined(&left)?;
        self.undefined_behavior.assert_value_not_undefined(&right)?;
        Ok(Value::from(left != right))
    }
    fn eq(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.undefined_behavior.assert_value_not_undefined(&left)?;
        self.undefined_behavior.assert_value_not_undefined(&right)?;
        Ok(Value::from(left == right))
    }
    fn order(&mut self, left: Value, right: Value, order: Order) -> Result<Value, Error> {
        self.undefined_behavior.assert_value_not_undefined(&left)?;
        self.undefined_behavior.assert_value_not_undefined(&right)?;
        Ok(Value::from(order.accepts(left.cmp(&right))))
    }
    fn contains(&mut self, container: Value, value: Value) -> Result<Value, Error> {
        self.state
            .undefined_behavior()
            .assert_iterable(&container)?;
        self.state
            .undefined_behavior()
            .assert_value_not_undefined(&value)?;
        ops::contains(&container, &value)
    }
    fn not(&mut self, value: Value) -> Result<Value, Error> {
        Ok(Value::from(!self.undefined_behavior.is_true(&value)?))
    }
    fn neg(&mut self, value: Value) -> Result<Value, Error> {
        ops::neg(&value)
    }
    fn perform_test(
        &mut self,
        name: &'env str,
        arguments: Option<u16>,
        local: u8,
    ) -> Result<(), Error> {
        let name = crate::vm::normalize_filter_test_name(name);
        let test = crate::vm::get_or_lookup_local(self.loaded_tests, local, || {
            self.state.env().get_test(name.as_ref())
        })
        .ok_or_else(|| {
            Error::new(
                ErrorKind::UnknownTest,
                format!("test {} is unknown", name.as_ref()),
            )
        })?;
        let arguments = self.stack.get_call_args(arguments);
        let count = arguments.len();
        let value = test.call(self.state, arguments)?;
        self.stack.drop_top(count);
        self.stack.push(Value::from(value.is_true()));
        Ok(())
    }
    fn truth(&self, value: &Value) -> Result<bool, Error> {
        self.undefined_behavior.is_true(value)
    }
    fn apply_filter(
        &mut self,
        name: &'env str,
        arguments: Option<u16>,
        local: u8,
    ) -> Result<(), Error> {
        let name = crate::vm::normalize_filter_test_name(name);
        let filter = crate::vm::get_or_lookup_local(self.loaded_filters, local, || {
            self.state.env().get_filter(name.as_ref())
        })
        .ok_or_else(|| {
            Error::new(
                ErrorKind::UnknownFilter,
                format!("filter {} is unknown", name.as_ref()),
            )
        })?;
        let arguments = self.stack.get_call_args(arguments);
        let count = arguments.len();
        let value = filter.call(self.state, arguments)?;
        self.stack.drop_top(count);
        self.stack.push(value);
        Ok(())
    }
    fn begin_capture(&mut self,mode:crate::output::CaptureMode)->Result<(),Error>{self.output.begin_capture(mode);Ok(())}
    fn end_capture(&mut self)->Result<Value,Error>{Ok(self.output.end_capture(self.state.auto_escape.get()))}
    fn emit(&mut self, value: Value) -> Result<(), Error> {
        if self.vm.env.is_default_formatter() {
            if self.strict_undefined
                && matches!(value.0, ValueRepr::Undefined(UndefinedType::Default))
            {
                return Err(Error::from(ErrorKind::UndefinedError));
            }
            write_escaped(self.output, self.state.auto_escape.get(), &value)
        } else {
            self.vm.env.format(&value, self.state, self.output)
        }
    }
    fn push_loop(&mut self, value: Value, flags: u8, pc: u32) -> Result<(), Error> {
        self.vm.push_loop(
            self.state,
            value,
            flags,
            pc,
            self.next_loop_recursion_jump.take(),
        )
    }
    fn next(&mut self) -> Result<Option<Value>, Error> {
        self.state
            .ctx
            .next_loop_item()
            .map(Value::validate)
            .transpose()
    }
    fn pop_loop(&mut self) -> Result<Option<u32>, Error> {
        let mut frame = self.state.ctx.pop_frame().current_loop.unwrap();
        if let Some((target, end_capture)) = frame.current_recursion_jump.take() {
            if end_capture {
                self.stack
                    .push(self.output.end_capture(self.state.auto_escape.get()));
            }
            Ok(Some(target))
        } else {
            Ok(None)
        }
    }
}
