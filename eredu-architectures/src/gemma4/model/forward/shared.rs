//! The ordinary policy table and its finite source-declared destination.
use super::{Metadata, retain};
use crate::gemma4::{ModelArgs, SharedAttentionStates, SharedAttentionStore};
use eredu_core::AttentionPolicy;
use eredu_nn::{AttentionStateSource, Error, workspace::WorkspaceContext};

/// Pass-local K/V publications. Counted forwards retain one mutable slot for
/// each actual declared publisher policy; ordinary forwards use the same map.
#[derive(Debug)]
pub struct SharedAttentionPublications<T> {
    values: Values<T>,
    // K/V values and their container are destroyed before the host account.
    metadata: Option<WorkspaceContext>,
}
#[derive(Debug)]
enum Values<T> {
    Ordinary(SharedAttentionStates<T>),
    Prepared(Vec<(AttentionPolicy, Option<(T,T)>)>),
}
impl<T> From<SharedAttentionStates<T>> for SharedAttentionPublications<T> {
    fn from(values: SharedAttentionStates<T>) -> Self {
        Self { values: Values::Ordinary(values), metadata: None }
    }
}
impl<T> SharedAttentionPublications<T> {
    pub(in crate::gemma4::model) fn prepare(args: &ModelArgs, metadata: Metadata<'_>) -> Result<Self, Error> {
        metadata.controls::<(Self, &ModelArgs, Option<WorkspaceContext>, usize,
            (AttentionPolicy, Option<(T,T)>))>()?;
        let Some(context) = metadata.context() else {
            return Ok(SharedAttentionStates::new().into());
        };
        let publisher = |index| args.layer_policy(index)
            .filter(|policy| matches!(policy.key_value, AttentionStateSource::Publish { .. }))
            .map(|policy| policy.attention);
        let selected = |index| publisher(index).filter(|policy|
            (0..index).all(|prior| publisher(prior).as_ref() != Some(policy)));
        let controls=std::mem::size_of_val(&publisher).checked_add(std::mem::size_of_val(&selected))
            .and_then(|n|n.checked_add(std::mem::size_of::<std::ops::Range<usize>>()))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(controls)?;
        let count = (0..args.num_hidden_layers()).filter_map(selected).count();
        let mut values = metadata.vector(count)?;
        for policy in (0..args.num_hidden_layers()).filter_map(selected) {
            values.push((policy,None));
        }
        Ok(Self { values: Values::Prepared(values), metadata: Some(context.clone()) })
    }

    pub(in crate::gemma4::model) fn prepare_policies(
        policies: &[(AttentionPolicy,i32,i32)], metadata: Metadata<'_>,
    ) -> Result<Self,Error> {
        metadata.controls::<(Self,&[(AttentionPolicy,i32,i32)],usize,
            (AttentionPolicy,Option<(T,T)>),Option<WorkspaceContext>)>()?;
        let Some(context)=metadata.context() else {return Ok(SharedAttentionStates::new().into());};
        let mut values=metadata.vector(policies.len())?;
        for (index,(policy,_,_)) in policies.iter().enumerate() {
            if policies[..index].iter().any(|(prior,_,_)|prior==policy) {
                return Err(retain(metadata,metadata.error(format_args!(
                    "Gemma boundary declared duplicate publisher policies"))));
            }
            values.push((*policy,None));
        }
        Ok(Self{values:Values::Prepared(values),metadata:Some(context.clone())})
    }
    pub(in crate::gemma4::model) fn contains_key(&self,policy:&AttentionPolicy)->bool {
        self.get(policy).is_some()
    }
    /// Borrows the currently published value for the exact attention policy.
    pub fn get(&self, policy: &AttentionPolicy) -> Option<&(T,T)> {
        match &self.values {
            Values::Ordinary(values) => values.get(policy),
            Values::Prepared(values) => values.iter().find(|(key,_)|key==policy)
                .and_then(|(_,value)|value.as_ref()),
        }
    }
    /// Iterates actual publications; unused declared slots are not observations.
    pub fn iter(&self) -> impl Iterator<Item=(&AttentionPolicy,&(T,T))> {
        let (ordinary,prepared) = match &self.values {
            Values::Ordinary(values) => (Some(values),None),
            Values::Prepared(values) => (None,Some(values)),
        };
        ordinary.into_iter().flat_map(|values|values.iter()).chain(
            prepared.into_iter().flat_map(|values|values.iter())
                .filter_map(|(policy,value)|value.as_ref().map(|value|(policy,value))))
    }
    /// Iterates each actual K/V pair without copying its container or tensors.
    pub fn values(&self) -> impl Iterator<Item=&(T,T)> { self.iter().map(|(_,value)|value) }
    /// Number of policies with an actual publication in this forward.
    pub fn len(&self) -> usize { self.iter().count() }
    /// Whether no layer has published a pair yet.
    pub fn is_empty(&self) -> bool { self.len()==0 }

    pub(in crate::gemma4::model) fn remove(&mut self, policy: &AttentionPolicy) -> Option<(T,T)> {
        match &mut self.values {
            Values::Ordinary(values) => values.remove(policy),
            Values::Prepared(values) => values.iter_mut().find(|(key,_)|key==policy)
                .and_then(|(_,value)|value.take()),
        }
    }
}
impl<T> SharedAttentionStore<T> for SharedAttentionPublications<T> {
    fn get(&self, policy: &AttentionPolicy) -> Option<&(T,T)> { self.get(policy) }
    fn publish(&mut self, policy: AttentionPolicy, value: (T,T)) -> Result<(),Error> {
        match &mut self.values {
            Values::Ordinary(values) => values.publish(policy,value),
            Values::Prepared(values) => {
                let metadata = Metadata::new(self.metadata.as_ref());
                metadata.controls::<(AttentionPolicy,(T,T),&mut Option<(T,T)>)>()?;
                let Some((_,slot))=values.iter_mut().find(|(key,_)|*key==policy) else {
                    return Err(retain(metadata,metadata.error(format_args!(
                        "Gemma K/V publication has no declared publisher policy"))));
                };
                *slot=Some(value);
                Ok(())
            }
        }
    }
}
