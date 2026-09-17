//! Paid borrowed-root destinations for the ordinary composite partition pass.
use crate::composite_execution::graph::Destination;
use eredu_nn::Error;

fn roots<'a,T,F>(destination:Destination<'_>,mut visit:F)->Result<Vec<&'a T>,Error>
where F:FnMut(&mut dyn FnMut(&'a T))->Result<(),Error> {
    destination.controls::<(F,Vec<&T>,Option<usize>,Option<Error>,usize)>()?;
    let mut count=Some(0usize);
    visit(&mut |_|count=count.and_then(|value|value.checked_add(1)))?;
    let count=count.ok_or_else(||destination.error(format_args!("composite root population overflow")))?;
    let mut values=destination.vector(count)?;
    let mut changed=false;
    visit(&mut |value|{
        if values.len()==count {changed=true;} else {values.push(value);}
    })?;
    if changed || values.len()!=count {
        return Err(destination.error(format_args!("composite root population changed during its source loan")));
    }
    Ok(values)
}
pub(super) fn state_values<'a,B,S>(state:&'a S,
    local:Option<(usize,eredu_runtime::ExecutionUnitAddress)>,destination:Destination<'_>)
    ->Result<Vec<&'a B::Tensor>,Error>
where B:eredu_nn::NeuralBackend,S:eredu_runtime::RuntimeState<B> {
    roots(destination,|visitor|match local {
        Some((local,address))=>state.visit_unit_retained_values(local,address,visitor)
            .map_err(|cause|super::pipeline_boundary::source(destination.0,cause)),
        None=>Ok(()),
    })
}
pub(super) fn context_values<'a,A,B,S>(architecture:&'a A,forward:&'a A::ForwardContext,
    group:usize,index:usize,destination:Destination<'_>)->Result<Vec<&'a B::Tensor>,Error>
where B:eredu_nn::NeuralBackend,S:eredu_runtime::RuntimeState<B>,A:eredu_runtime::LayeredArchitecture<B,S> {
    roots(destination,|visitor|{
        architecture.visit_retained_context_values(forward,group,index,visitor);Ok(())
    })
}
