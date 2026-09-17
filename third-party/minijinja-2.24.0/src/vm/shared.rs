//! Storage-parametric dispatch for the selected instructions and filter calls.
//!
//! Ordinary evaluation and closed source-bound evaluation use this same operand
//! ordering and control flow. Storage decides how a value is represented; it
//! cannot replace this instruction dispatcher.

#![forbid(unsafe_code)]

use crate::compiler::instructions::Instruction;
use crate::value::Value;
use crate::value::primitive::scalar::Arithmetic;
pub(crate) mod macro_arguments;

/// The ordinary ordering predicates, independent of value storage.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Order {
    Lt,
    Lte,
    Gt,
    Gte,
}
impl Order {
    pub(crate) fn accepts(self, ordering: std::cmp::Ordering) -> bool {
        match self {
            Self::Lt => ordering.is_lt(),
            Self::Lte => !ordering.is_gt(),
            Self::Gt => ordering.is_gt(),
            Self::Gte => !ordering.is_lt(),
        }
    }
}

pub(crate) enum Op<'source, V> {
    Lookup(&'source str),
    StoreLocal(&'source str),
    SetAttr(&'source str),
    BuildKwargs(usize),
    BuildList(usize),
    BeginCollection,
    AppendCollection,
    EndCollection,
    BuildMap(usize),
    CallFunction(&'source str, Option<u16>, Option<V>),
    DupTop,
    DiscardTop,
    Enclose(&'source str),
    GetClosure,
    BuildMacro(&'source str, u32, u8),
    Return,
    UnpackList(usize),
    GetAttr(&'source str),
    GetItem,
    Slice,
    CallMethod(&'source str, Option<u16>),
    LoadConst(V),
    Add,
    StringConcat,
    Arithmetic(Arithmetic),
    Ne,
    Eq,
    Order(Order),
    In,
    Not,
    Neg,
    PerformTest(&'source str, Option<u16>, u8),
    ApplyFilter(&'source str, Option<u16>, u8),
    BeginCapture(crate::output::CaptureMode),
    EndCapture,
    Emit,
    PushLoop(u8),
    Iterate(u32),
    PopLoopFrame,
    Jump(u32),
    JumpIfFalse(u32),
    JumpIfFalseOrPop(u32),
    JumpIfTrueOrPop(u32),
}

/// No source/allocator callback is exposed by the closed public API. This is the
/// private storage contract used by the existing ordinary VM as well.
pub(crate) trait Storage<'source> {
    type Value;
    type Error;
    fn pop(&mut self) -> Result<Self::Value, Self::Error>;
    fn peek(&self) -> Result<&Self::Value, Self::Error>;
    fn push(&mut self, value: Self::Value) -> Result<(), Self::Error>;
    fn lookup(&mut self, name: &'source str) -> Result<Self::Value, Self::Error>;
    fn store(&mut self, name: &'source str, value: Self::Value) -> Result<(), Self::Error>;
    fn set_attr(
        &mut self,
        object: Self::Value,
        name: &'source str,
        value: Self::Value,
    ) -> Result<(), Self::Error>;
    fn build_kwargs(&mut self, count: usize) -> Result<(), Self::Error>;
    fn build_list(&mut self, count: usize) -> Result<(), Self::Error>;
    fn begin_collection(&mut self) -> Result<(), Self::Error>;
    fn append_collection(&mut self, value: Self::Value) -> Result<(), Self::Error>;
    fn end_collection(&mut self) -> Result<Self::Value, Self::Error>;
    fn build_map(&mut self, count: usize) -> Result<(), Self::Error>;
    fn call_function(
        &mut self,
        name: &'source str,
        arguments: Option<u16>,
        function: Option<Self::Value>,
        pc: u32,
    ) -> Result<Option<u32>, Self::Error>;
    fn duplicate(&mut self) -> Result<(), Self::Error>;
    fn enclose(&mut self, name: &'source str) -> Result<(), Self::Error>;
    fn closure(&mut self) -> Result<Self::Value, Self::Error>;
    fn build_macro(
        &mut self,
        name: &'source str,
        offset: u32,
        flags: u8,
    ) -> Result<(), Self::Error>;
    fn return_macro(&mut self) -> Result<Option<u32>, Self::Error>;
    fn unpack(&mut self, count: usize) -> Result<(), Self::Error>;
    fn attr(&mut self, value: Self::Value, name: &str) -> Result<Self::Value, Self::Error>;
    fn item(&mut self, value: Self::Value, key: Self::Value) -> Result<Self::Value, Self::Error>;
    fn slice(&mut self, value: Self::Value, start: Self::Value, stop: Self::Value, step: Self::Value) -> Result<Self::Value, Self::Error>;
    fn call_method(
        &mut self,
        name: &'source str,
        arguments: Option<u16>,
    ) -> Result<(), Self::Error>;
    fn add(&mut self, left: Self::Value, right: Self::Value) -> Result<Self::Value, Self::Error>;
    fn string_concat(&mut self, left: Self::Value, right: Self::Value) -> Result<Self::Value, Self::Error>;
    fn arithmetic(
        &mut self,
        operation: Arithmetic,
        left: Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    fn ne(&mut self, left: Self::Value, right: Self::Value) -> Result<Self::Value, Self::Error>;
    fn eq(&mut self, left: Self::Value, right: Self::Value) -> Result<Self::Value, Self::Error>;
    fn order(
        &mut self,
        left: Self::Value,
        right: Self::Value,
        order: Order,
    ) -> Result<Self::Value, Self::Error>;
    fn contains(
        &mut self,
        container: Self::Value,
        value: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    fn not(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn neg(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn perform_test(
        &mut self,
        name: &'source str,
        arguments: Option<u16>,
        local: u8,
    ) -> Result<(), Self::Error>;
    fn truth(&self, value: &Self::Value) -> Result<bool, Self::Error>;
    fn apply_filter(
        &mut self,
        name: &'source str,
        arguments: Option<u16>,
        local: u8,
    ) -> Result<(), Self::Error>;
    fn begin_capture(&mut self, mode:crate::output::CaptureMode)->Result<(),Self::Error>;
    fn end_capture(&mut self)->Result<Self::Value,Self::Error>;
    fn emit(&mut self, value: Self::Value) -> Result<(), Self::Error>;
    fn push_loop(&mut self, value: Self::Value, flags: u8, pc: u32) -> Result<(), Self::Error>;
    fn next(&mut self) -> Result<Option<Self::Value>, Self::Error>;
    fn pop_loop(&mut self) -> Result<Option<u32>, Self::Error>;
}

pub(crate) enum Next {
    Advance,
    Jump(u32),
    Return,
}

pub(crate) fn step<'source, S: Storage<'source>>(
    op: Op<'source, S::Value>,
    pc: u32,
    storage: &mut S,
) -> Result<Next, S::Error> {
    match op {
        Op::Lookup(name) => {
            let value = storage.lookup(name)?;
            storage.push(value)?;
        }
        Op::StoreLocal(name) => {
            let value = storage.pop()?;
            storage.store(name, value)?;
        }
        Op::SetAttr(name) => {
            let object = storage.pop()?;
            let value = storage.pop()?;
            storage.set_attr(object, name, value)?;
        }
        Op::BuildKwargs(count) => storage.build_kwargs(count)?,
        Op::BuildList(count) => storage.build_list(count)?,
        Op::BuildMap(count) => storage.build_map(count)?,
        Op::CallFunction(name, arguments, function) => {
            if let Some(target) = storage.call_function(name, arguments, function, pc)? {
                return Ok(Next::Jump(target));
            }
        }
        Op::DupTop => storage.duplicate()?,
        Op::DiscardTop => {
            storage.pop()?;
        }
        Op::Enclose(name) => storage.enclose(name)?,
        Op::GetClosure => {
            let value = storage.closure()?;
            storage.push(value)?;
        }
        Op::BuildMacro(name, offset, flags) => storage.build_macro(name, offset, flags)?,
        Op::Return => {
            return Ok(match storage.return_macro()? {
                Some(target) => Next::Jump(target),
                None => Next::Return,
            });
        }
        Op::UnpackList(count) => storage.unpack(count)?,
        Op::GetAttr(name) => {
            let value = storage.pop()?;
            let value = storage.attr(value, name)?;
            storage.push(value)?;
        }
        Op::GetItem => {
            let key = storage.pop()?;
            let value = storage.pop()?;
            let value = storage.item(value, key)?;
            storage.push(value)?;
        }
        Op::Slice => {
            let step = storage.pop()?;
            let stop = storage.pop()?;
            let start = storage.pop()?;
            let value = storage.pop()?;
            let value = storage.slice(value, start, stop, step)?;
            storage.push(value)?;
        }
        Op::CallMethod(name, arguments) => storage.call_method(name, arguments)?,
        Op::LoadConst(value) => storage.push(value)?,
        Op::Add => {
            let right = storage.pop()?;
            let left = storage.pop()?;
            let value = storage.add(left, right)?;
            storage.push(value)?;
        }
        Op::StringConcat => {
            let right = storage.pop()?;
            let left = storage.pop()?;
            let value = storage.string_concat(left, right)?;
            storage.push(value)?;
        }
        Op::Arithmetic(operation) => {
            let right = storage.pop()?;
            let left = storage.pop()?;
            let value = storage.arithmetic(operation, left, right)?;
            storage.push(value)?;
        }
        Op::Ne => {
            let right = storage.pop()?;
            let left = storage.pop()?;
            let value = storage.ne(left, right)?;
            storage.push(value)?;
        }
        Op::Eq => {
            let right = storage.pop()?;
            let left = storage.pop()?;
            let value = storage.eq(left, right)?;
            storage.push(value)?;
        }
        Op::Order(order) => {
            let right = storage.pop()?;
            let left = storage.pop()?;
            let value = storage.order(left, right, order)?;
            storage.push(value)?;
        }
        Op::In => {
            let container = storage.pop()?;
            let value = storage.pop()?;
            let result = storage.contains(container, value)?;
            storage.push(result)?;
        }
        Op::Not => {
            let value = storage.pop()?;
            let value = storage.not(value)?;
            storage.push(value)?;
        }
        Op::Neg => {
            let value = storage.pop()?;
            let value = storage.neg(value)?;
            storage.push(value)?;
        }
        Op::PerformTest(name, arguments, local) => storage.perform_test(name, arguments, local)?,
        Op::ApplyFilter(name, arguments, local) => storage.apply_filter(name, arguments, local)?,
        Op::BeginCollection=>storage.begin_collection()?,
        Op::AppendCollection=>{let value=storage.pop()?;storage.append_collection(value)?;}
        Op::EndCollection=>{let value=storage.end_collection()?;storage.push(value)?;}
        Op::BeginCapture(mode)=>storage.begin_capture(mode)?,
        Op::EndCapture=>{let value=storage.end_capture()?;storage.push(value)?;}
        Op::Emit => {
            let value = storage.pop()?;
            storage.emit(value)?;
        }
        Op::PushLoop(flags) => {
            let value = storage.pop()?;
            storage.push_loop(value, flags, pc)?;
        }
        Op::Iterate(target) => match storage.next()? {
            Some(value) => storage.push(value)?,
            None => return Ok(Next::Jump(target)),
        },
        Op::PopLoopFrame => {
            if let Some(target) = storage.pop_loop()? {
                return Ok(Next::Jump(target));
            }
        }
        Op::Jump(target) => return Ok(Next::Jump(target)),
        Op::JumpIfFalse(target) => {
            let value = storage.pop()?;
            if !storage.truth(&value)? {
                return Ok(Next::Jump(target));
            }
        }
        Op::JumpIfFalseOrPop(target) => {
            if !storage.truth(storage.peek()?)? {
                return Ok(Next::Jump(target));
            }
            storage.pop()?;
        }
        Op::JumpIfTrueOrPop(target) => {
            if storage.truth(storage.peek()?)? {
                return Ok(Next::Jump(target));
            }
            storage.pop()?;
        }
    }
    Ok(Next::Advance)
}

pub(super) fn ordinary<'source>(instruction: &Instruction<'source>) -> Option<Op<'source, Value>> {
    Some(match instruction {
        Instruction::Lookup(name) => Op::Lookup(name),
        Instruction::DupTop => Op::DupTop,
        Instruction::DiscardTop => Op::DiscardTop,
        #[cfg(feature = "macros")]
        Instruction::Enclose(name) => Op::Enclose(name),
        #[cfg(feature = "macros")]
        Instruction::GetClosure => Op::GetClosure,
        #[cfg(feature = "macros")]
        Instruction::BuildMacro(name, offset, flags) => Op::BuildMacro(name, *offset, *flags),
        #[cfg(feature = "macros")]
        Instruction::Return => Op::Return,
        Instruction::StoreLocal(name) => Op::StoreLocal(name),
        Instruction::SetAttr(name) => Op::SetAttr(name),
        Instruction::BuildKwargs(count) => Op::BuildKwargs(*count),
        Instruction::BuildList(Some(count)) => Op::BuildList(*count),
        Instruction::BeginCollection=>Op::BeginCollection,
        Instruction::AppendCollection=>Op::AppendCollection,
        Instruction::EndCollection=>Op::EndCollection,
        Instruction::BuildMap(count) => Op::BuildMap(*count),
        Instruction::UnpackList(count) => Op::UnpackList(*count),
        Instruction::GetAttr(name) => Op::GetAttr(name),
        Instruction::GetItem => Op::GetItem,
        Instruction::Slice => Op::Slice,
        Instruction::CallMethod(name, arguments) => Op::CallMethod(name, *arguments),
        Instruction::LoadConst(value) => Op::LoadConst(value.clone()),
        Instruction::Add => Op::Add,
        Instruction::StringConcat => Op::StringConcat,
        Instruction::Sub => Op::Arithmetic(Arithmetic::Sub),
        Instruction::Mul => Op::Arithmetic(Arithmetic::Mul),
        Instruction::Div => Op::Arithmetic(Arithmetic::Div),
        Instruction::IntDiv => Op::Arithmetic(Arithmetic::FloorDiv),
        Instruction::Rem => Op::Arithmetic(Arithmetic::Rem),
        Instruction::Pow => Op::Arithmetic(Arithmetic::Pow),
        Instruction::Ne => Op::Ne,
        Instruction::Eq => Op::Eq,
        Instruction::In => Op::In,
        Instruction::Lt => Op::Order(Order::Lt),
        Instruction::Lte => Op::Order(Order::Lte),
        Instruction::Gt => Op::Order(Order::Gt),
        Instruction::Gte => Op::Order(Order::Gte),
        Instruction::Not => Op::Not,
        Instruction::Neg => Op::Neg,
        Instruction::PerformTest(name, args, local) => Op::PerformTest(name, *args, *local),
        Instruction::ApplyFilter(name, arguments, local) => {
            Op::ApplyFilter(name, *arguments, *local)
        }
        Instruction::BeginCapture(mode)=>Op::BeginCapture(*mode),
        Instruction::EndCapture=>Op::EndCapture,
        Instruction::Emit => Op::Emit,
        Instruction::PushLoop(flags) => Op::PushLoop(*flags),
        Instruction::Iterate(target) => Op::Iterate(*target),
        Instruction::PopLoopFrame => Op::PopLoopFrame,
        Instruction::Jump(target) => Op::Jump(*target),
        Instruction::JumpIfFalse(target) => Op::JumpIfFalse(*target),
        Instruction::JumpIfFalseOrPop(target) => Op::JumpIfFalseOrPop(*target),
        Instruction::JumpIfTrueOrPop(target) => Op::JumpIfTrueOrPop(*target),
        _ => return None,
    })
}

pub(super) mod ordinary;
