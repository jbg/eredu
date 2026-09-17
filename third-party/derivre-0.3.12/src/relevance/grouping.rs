//! One stable first-occurrence grouping worker for symbolic selectors/values.
pub(crate) mod storage;
use super::SymRes;
use crate::{
    HashMap,
    ast::{ExprRef, ExprSet},
};
use std::convert::Infallible;

pub(crate) trait Memory {
    type Error;
    fn begin(&mut self) -> Result<(), Self::Error>;
    fn get(&self, key: ExprRef) -> Option<usize>;
    fn index(&mut self, key: ExprRef, index: usize) -> Result<(), Self::Error>;
    fn clear_index(&mut self);
    fn groups(&self) -> usize;
    fn add_group(&mut self, key: ExprRef, value: ExprRef) -> Result<(), Self::Error>;
    fn append(&mut self, index: usize, value: ExprRef) -> Result<(), Self::Error>;
    fn group(&mut self, index: usize) -> (ExprRef, &mut Vec<ExprRef>);
    fn release_group(&mut self, index: usize);
    fn output(&mut self, count: usize) -> Result<SymRes, Self::Error>;
    fn push(&mut self, out: &mut SymRes, value: (ExprRef, ExprRef)) -> Result<(), Self::Error>;
    fn retire_input(&mut self, input: SymRes);
}
pub(crate) fn group<M: Memory>(
    input: SymRes,
    memory: &mut M,
    mut combine: impl FnMut(&mut Vec<ExprRef>) -> Result<ExprRef, M::Error>,
) -> Result<SymRes, M::Error> {
    memory.begin()?;
    let mut duplicate = false;
    for &(key, _) in &input {
        if memory.get(key).is_some() {
            duplicate = true;
        } else {
            memory.index(key, 0)?;
        }
    }
    if !duplicate {
        return Ok(input);
    }
    memory.clear_index();
    for &(key, value) in &input {
        if let Some(index) = memory.get(key) {
            memory.append(index, value)?;
        } else {
            let index = memory.groups();
            memory.index(key, index)?;
            memory.add_group(key, value)?;
        }
    }
    let mut output = memory.output(memory.groups())?;
    for index in 0..memory.groups() {
        let (key, values) = memory.group(index);
        let value = if values.len() == 1 {
            values[0]
        } else {
            combine(values)?
        };
        memory.push(&mut output, (key, value))?;
        memory.release_group(index);
    }
    memory.retire_input(input);
    Ok(output)
}
pub(crate) trait Combine {
    type Error;
    fn pay(&mut self, amount: usize) -> Result<(), Self::Error>;
    fn regex(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error>;
    fn selectors(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error>;
}
pub(crate) fn simplify<C: Combine, M: Memory<Error = C::Error>>(
    input: SymRes,
    memory: &mut M,
    combine: &mut C,
) -> Result<SymRes, C::Error> {
    if input.len() <= 1 || (input.len() == 2 && input[0].0 != input[1].0) {
        return Ok(input);
    }
    combine.pay(input.len())?;
    let mut grouped = group(input, memory, |args| combine.regex(args))?;
    super::swap_each(&mut grouped);
    let mut grouped = group(grouped, memory, |args| combine.selectors(args))?;
    super::swap_each(&mut grouped);
    Ok(grouped)
}
#[derive(Default)]
struct OrdinaryMemory {
    index: HashMap<ExprRef, usize>,
    groups: Vec<(ExprRef, Vec<ExprRef>)>,
}
impl Memory for OrdinaryMemory {
    type Error = Infallible;
    fn begin(&mut self) -> Result<(), Infallible> {
        *self = Self::default();
        Ok(())
    }
    fn get(&self, key: ExprRef) -> Option<usize> {
        self.index.get(&key).copied()
    }
    fn index(&mut self, key: ExprRef, index: usize) -> Result<(), Infallible> {
        self.index.insert(key, index);
        Ok(())
    }
    fn clear_index(&mut self) {
        self.index.clear();
    }
    fn groups(&self) -> usize {
        self.groups.len()
    }
    fn add_group(&mut self, key: ExprRef, value: ExprRef) -> Result<(), Infallible> {
        self.groups.push((key, vec![value]));
        Ok(())
    }
    fn append(&mut self, index: usize, value: ExprRef) -> Result<(), Infallible> {
        self.groups[index].1.push(value);
        Ok(())
    }
    fn group(&mut self, index: usize) -> (ExprRef, &mut Vec<ExprRef>) {
        let (key, args) = &mut self.groups[index];
        (*key, args)
    }
    fn release_group(&mut self, index: usize) {
        drop(std::mem::take(&mut self.groups[index].1));
    }
    fn output(&mut self, count: usize) -> Result<SymRes, Infallible> {
        Ok(Vec::with_capacity(count))
    }
    fn push(&mut self, out: &mut SymRes, value: (ExprRef, ExprRef)) -> Result<(), Infallible> {
        out.push(value);
        Ok(())
    }
    fn retire_input(&mut self, _input: SymRes) {}
}
struct Ordinary<'a>(&'a mut ExprSet);
impl Combine for Ordinary<'_> {
    type Error = Infallible;
    fn pay(&mut self, amount: usize) -> Result<(), Infallible> {
        self.0.pay(amount);
        Ok(())
    }
    fn regex(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Infallible> {
        Ok(self.0.mk_or(args))
    }
    fn selectors(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Infallible> {
        Ok(self.0.mk_byte_set_or(args))
    }
}
pub(super) fn ordinary_simplify(source: &mut ExprSet, input: SymRes) -> SymRes {
    match simplify(input, &mut OrdinaryMemory::default(), &mut Ordinary(source)) {
        Ok(result) => result,
        Err(never) => match never {},
    }
}
