//! Row movement ceilings derived from the selected region and local outputs.
use super::*;
use crate::backend::nn::expert_movement;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum ExpertMovementKind { Gather { route_values:bool }, Zero { rows:i32 }, Add }

#[derive(Clone, Copy, Debug)]
struct MovementInput {
    rank: usize,
    leading: [i32; 2],
    dtype: safemlx::Dtype,
}
#[derive(Debug, thiserror::Error)]
#[error("actual expert movement {kind:?} failed {site}: count={count:?}, inputs={inputs:?}; native [graph,records,backing,births,bytes,controls,kernels] actual={actual:?} ceiling={ceiling:?}")]
struct MovementMismatch {
    kind: ExpertMovementKind,
    site: &'static str,
    count: Option<usize>,
    inputs: [Option<MovementInput>; 3],
    actual: Option<[u64; 7]>,
    ceiling: Option<[u64; 7]>,
}

fn inputs(native: &[crate::MlxTensor]) -> [Option<MovementInput>; 3] {
    std::array::from_fn(|index| native.get(index).map(|value| {
        let shape = value.shape();
        MovementInput { rank: shape.len(), leading: std::array::from_fn(|axis|
            shape.get(axis).copied().unwrap_or(0)), dtype: value.as_array().dtype() }
    }))
}

fn metrics(capacity: BoundaryStageCapacity, births: usize, bytes: u64, controls: u64,
    kernels: usize) -> [u64; 7] {
    // Diagnostic counters only; saturation never grants a construction bound.
    let wide = |value| u64::try_from(value).unwrap_or(u64::MAX);
    [wide(capacity.graph), wide(capacity.records), wide(capacity.backing),
        wide(births), bytes, controls, wide(kernels)]
}
struct Profile {
    kind:ExpertMovementKind,
    base:WorkspaceLayout,
    updates:Option<WorkspaceLayout>,
    maximum_indices:usize,
    capacity:BoundaryStageCapacity,
    births:usize,
    bytes:u64,
    controls:u64,
    kernels:usize,
}
pub(crate) struct ExpertMovementQuote {
    profiles:Vec<Profile>,
    mechanism:ResidentExecutionMechanisms,
    source:OriginalParallelSource,
}
impl ExpertMovementQuote {
    pub(super) fn prepare(local:&ExpertLocalQuote)->Result<Self,Error> {
        let source=&local.source;
        let context=WorkspaceContext::new_with_metadata_funding(local.mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,Profile,Vec<Profile>,Result<Self,Error>,WorkspaceContext)>())?;
        let invalid=||context.metadata_error(format_args!("expert movement lacks its retained numerical source"));
        let declaration=local.declaration.as_view();
        let selected=declaration.selected_rows().ok_or_else(invalid)?;
        let selected_i32=i32::try_from(selected).map_err(|_|invalid())?;
        let rows=i32::try_from(declaration.source_rows).map_err(|_|invalid())?;
        let routes=i32::try_from(declaration.routes_per_row).map_err(|_|invalid())?;
        let shape=|prototype:&WorkspaceLayout,shape:&[i32]|->Result<WorkspaceLayout,Error>{
            if prototype.representation().is_none(){return Err(invalid());}
            Ok(context.layout(shape,prototype.dtype())?.with_representation(prototype.representation()))
        };
        let mut profiles=context.metadata_vec(3+local.outputs.len().checked_mul(3).ok_or_else(invalid)?)?;
        profiles.push(Self::profile(ExpertMovementKind::Gather{route_values:false},
            shape(&local.inputs[0],&[rows,declaration.kernel.dimensions().0])?,None,selected,local)?);
        for scalar in &local.inputs[2..4] {
            profiles.push(Self::profile(ExpertMovementKind::Gather{route_values:true},
                shape(scalar,&[rows,routes])?,None,selected,local)?);
        }
        for output in &local.outputs {
            let returned=shape(output,&[selected_i32,declaration.kernel.dimensions().1])?;
            profiles.push(Self::profile(ExpertMovementKind::Gather{route_values:false},returned.clone(),None,selected,local)?);
            profiles.push(Self::profile(ExpertMovementKind::Zero{rows},returned.clone(),None,0,local)?);
            profiles.push(Self::profile(ExpertMovementKind::Add,
                shape(output,&[rows,declaration.kernel.dimensions().1])?,Some(returned),selected,local)?);
        }
        Ok(Self{profiles,mechanism:local.mechanism,source:source.clone()})
    }
    fn profile(kind:ExpertMovementKind,base:WorkspaceLayout,updates:Option<WorkspaceLayout>,maximum:usize,
        local:&ExpertLocalQuote)->Result<Profile,Error> {
        let context=WorkspaceContext::new_with_metadata_funding(local.mechanism,local.source.funding().clone())?;
        context.charge_metadata(size_of::<(Profile,Result<Profile,Error>,[usize;3],Option<Profile>)>())?;
        let invalid=||context.metadata_error(format_args!("expert movement profile has no native worker"));
        let mut profile=Profile{kind,base,updates,maximum_indices:maximum,
            capacity:BoundaryStageCapacity{graph:0,records:0,backing:0},births:0,bytes:0,controls:0,kernels:0};
        // These workers have one zero/singleton branch; positive row storage,
        // checked gather validation and rank-two ScatterAxis populations are
        // componentwise monotone. Shapes/widths and dtype stay fixed here.
        for count in [0,1,maximum] {
            if count>maximum{continue;}
            let recipe=trace(&profile,count,local.mechanism,local.source.funding())?;
            let capacity=boundary::capacity(recipe,local.source.agreement_inputs().ok_or_else(invalid)?.runtime(),&context)?;
            profile.capacity.graph=profile.capacity.graph.max(capacity.graph);
            profile.capacity.records=profile.capacity.records.max(capacity.records);
            profile.capacity.backing=profile.capacity.backing.max(capacity.backing);
            profile.births=profile.births.max(recipe.storage.maximum_births());
            profile.bytes=profile.bytes.max(recipe.storage.mutable_bytes());
            profile.controls=profile.controls.max(recipe.controls);profile.kernels=profile.kernels.max(recipe.kernels);
        }
        Ok(profile)
    }
    pub(crate) fn actual(&self,kind:ExpertMovementKind,native:&[crate::MlxTensor])
        ->Result<(SpeculativeNumericalRecipe,BoundaryStageCapacity),Error> {
        let context=WorkspaceContext::new_with_metadata_funding(self.mechanism,self.source.funding().clone())?;
        context.charge_metadata(size_of::<(WorkspaceContext,Result<(SpeculativeNumericalRecipe,BoundaryStageCapacity),Error>,Vec<WorkspaceLayout>,
            MovementMismatch, MovementMismatch, [Option<MovementInput>; 3], [u64; 7], [u64; 7])>())?;
        let mismatch = |site, count, actual, ceiling| context.metadata_source(MovementMismatch {
            kind, site, count, inputs: inputs(native), actual, ceiling,
        });
        let invalid=||mismatch("input declaration",None,None,None);
        let count=match kind {ExpertMovementKind::Zero{..}=>0,_=>usize::try_from(*native.get(1).ok_or_else(invalid)?.shape().first().ok_or_else(invalid)?).map_err(|_|invalid())?};
        let matches=|profile:&&Profile| {
            if profile.kind!=kind||count>profile.maximum_indices||native.first().is_none_or(|value|value.shape()!=profile.base.shape()){return false;}
            let float=|value:&crate::MlxTensor,layout:&WorkspaceLayout| {
                crate::backend::nn::workspace::byte_view::Dtype::from_layout(layout.as_view())
                    .is_some_and(|dtype|dtype.native()==value.as_array().dtype())
            };
            if !float(&native[0],&profile.base){return false;}
            match (kind,&profile.updates) {
                (ExpertMovementKind::Gather{..},None)=>native.len()==2&&native[1].shape()==[count as i32]&&native[1].as_array().dtype()==safemlx::Dtype::Int32,
                (ExpertMovementKind::Zero{..},None)=>native.len()==1,
                (ExpertMovementKind::Add,Some(updates))=>native.len()==3&&native[1].shape()==[count as i32,1]
                    &&native[1].as_array().dtype()==safemlx::Dtype::Int32&&native[2].shape()==[count as i32,updates.shape()[1]]&&float(&native[2],updates),
                _=>false,
            }
        };
        let profile=self.profiles.iter().find(matches).ok_or_else(||
            mismatch("profile shape/dtype",Some(count),None,None))?;
        let mut projection=ExistingArrayProjection::with_source_count(&context,native.len())
            .map_err(|cause|context.metadata_source(cause))?;
        let mut projected=context.metadata_vec(native.len())?;
        for value in native {projected.push(projection.project(value.as_array())?);}
        if !projection.is_complete(){return Err(mismatch("native input projection",Some(count),None,None));}
        let actual=Profile{kind,base:projected[0].layout().clone(),
            updates:projected.get(2).map(|value|value.layout().clone()),maximum_indices:count,
            capacity:profile.capacity,births:profile.births,bytes:profile.bytes,controls:profile.controls,kernels:profile.kernels};
        let recipe=trace(&actual,count,self.mechanism,self.source.funding())?;
        let capacity=boundary::capacity(recipe,self.source.agreement_inputs().ok_or_else(invalid)?.runtime(),&context)?;
        if capacity.graph>profile.capacity.graph||capacity.records>profile.capacity.records||capacity.backing>profile.capacity.backing
            ||recipe.storage.maximum_births()>profile.births||recipe.storage.mutable_bytes()>profile.bytes
            ||recipe.controls>profile.controls||recipe.kernels>profile.kernels{
            return Err(mismatch("recipe ceiling",Some(count),
                Some(metrics(capacity,recipe.storage.maximum_births(),recipe.storage.mutable_bytes(),recipe.controls,recipe.kernels)),
                Some(metrics(profile.capacity,profile.births,profile.bytes,profile.controls,profile.kernels))));
        }
        Ok((recipe,profile.capacity))
    }
    pub(super) fn per_kind_population(&self,index:usize)->Option<(usize,usize)> {
        let mut found=false;let mut backing=0;let mut births=0;
        for p in &self.profiles {
            let slot=match p.kind{ExpertMovementKind::Gather{route_values:false}=>0,
                ExpertMovementKind::Gather{route_values:true}=>1,ExpertMovementKind::Zero{..}=>2,ExpertMovementKind::Add=>3};
            if slot==index{found=true;backing=backing.max(p.capacity.backing);births=births.max(p.births);}
        }
        found.then_some((backing,births))
    }
    pub(crate) fn capacities(&self)->impl Iterator<Item=(BoundaryStageCapacity,usize,u64,u64,usize)>+'_ {
        self.profiles.iter().map(|p|(p.capacity,p.births,p.bytes,p.controls,p.kernels))
    }
}
fn trace(profile:&Profile,count:usize,mechanism:ResidentExecutionMechanisms,funding:&HostMetadataFunding)
    ->Result<SpeculativeNumericalRecipe,Error> {
    let context=WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone())?;
    context.charge_metadata(size_of::<(WorkspaceContext,WorkspaceTraceReport,[WorkspaceTensor;3],Result<SpeculativeNumericalRecipe,Error>)>())?;
    let invalid=||context.metadata_error(format_args!("movement trace exceeds retained row geometry"));
    let base=WorkspaceTensor::existing(context.layout(profile.base.shape(),profile.base.dtype())?
        .with_representation(profile.base.representation()),&context)?;
    let count=i32::try_from(count).map_err(|_|invalid())?;
    let index=match profile.kind {
        ExpertMovementKind::Zero{..}=>None,
        ExpertMovementKind::Gather{..}=>Some(WorkspaceTensor::existing(context.layout(&[count],WorkspaceDtype::Int32)?,&context)?),
        ExpertMovementKind::Add=>Some(WorkspaceTensor::existing(context.layout(&[count,1],WorkspaceDtype::Int32)?,&context)?),
    };
    let updates=profile.updates.as_ref().map(|value|WorkspaceTensor::existing(
        context.layout(&[count,value.shape()[1]],value.dtype())?.with_representation(value.representation()),&context)).transpose()?;
    context.begin_span();let ops=expert_movement::Workspace(&context);
    let output=match profile.kind {
        ExpertMovementKind::Gather{route_values}=>expert_movement::gather(&ops,&base,index.as_ref().ok_or_else(invalid)?,route_values)?,
        ExpertMovementKind::Zero{rows}=>expert_movement::zeros(&ops,&base,rows)?,
        ExpertMovementKind::Add=>expert_movement::add(&ops,&base,index.as_ref().ok_or_else(invalid)?,updates.as_ref().ok_or_else(invalid)?)?,
    };
    let report=context.finish_report(&[output])?;
    SpeculativeNumericalRecipe::inspect_owned_child(&report,1,mechanism,&context)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
