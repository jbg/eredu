//! Immutable source-funded K/V rows use the same policy lookup as ordinary maps.
use super::super::{SharedAttentionStates, SharedAttentionStore};
use eredu_core::{AttentionPolicy, SpeculativeValues};

/// Shared target captures for an assistant branch. Prepared rows share their
/// existing owner; ordinary state preserves HashMap clone/publication behavior.
#[derive(Debug, Clone)]
pub struct SharedAssistantStates<T>(Storage<T>);
#[derive(Debug, Clone)]
enum Storage<T> { Ordinary(SharedAttentionStates<T>), Prepared(SpeculativeValues<(AttentionPolicy, (T,T))>) }
impl<T> SharedAssistantStates<T> {
    /// Uses the immutable rows born in the actual paid state constructor.
    pub fn from_prepared(values: SpeculativeValues<(AttentionPolicy, (T,T))>) -> Self { Self(Storage::Prepared(values)) }
    /// Number of actual policy entries.
    pub fn len(&self)->usize {match &self.0 {Storage::Ordinary(v)=>v.len(),Storage::Prepared(v)=>v.len()}}
    /// Whether this context has no policy entries.
    pub fn is_empty(&self)->bool {self.len()==0}
    /// Iterates the actual retained declaration rows.
    pub fn iter(&self)->Iter<'_,T> {match &self.0 {Storage::Ordinary(v)=>Iter(Iteration::Ordinary(v.iter())),Storage::Prepared(v)=>Iter(Iteration::Prepared(v.iter()))}}
    /// Borrows every actual K/V pair.
    pub fn values(&self)->impl ExactSizeIterator<Item=&(T,T)> + Clone {self.iter().map(|(_,v)|v)}
}
impl<T> Default for SharedAssistantStates<T> {fn default()->Self{SharedAttentionStates::new().into()}}
impl<T> From<SharedAttentionStates<T>> for SharedAssistantStates<T> {fn from(v:SharedAttentionStates<T>)->Self{Self(Storage::Ordinary(v))}}
impl<T> FromIterator<(AttentionPolicy,(T,T))> for SharedAssistantStates<T> {fn from_iter<I:IntoIterator<Item=(AttentionPolicy,(T,T))>>(i:I)->Self{Self(Storage::Ordinary(i.into_iter().collect()))}}
impl<T> SharedAttentionStore<T> for SharedAssistantStates<T> {
    fn get(&self,p:&AttentionPolicy)->Option<&(T,T)>{match &self.0 {Storage::Ordinary(v)=>v.get(p),Storage::Prepared(v)=>v.iter().find(|(k,_)|k==p).map(|(_,v)|v)}}
    fn publish(&mut self,p:AttentionPolicy,v:(T,T))->Result<(),eredu_nn::Error>{match &mut self.0 {
        Storage::Ordinary(values)=>values.publish(p,v),
        Storage::Prepared(_)=>Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into()),
    }}
}
impl<T> std::ops::Index<&AttentionPolicy> for SharedAssistantStates<T> {type Output=(T,T);fn index(&self,p:&AttentionPolicy)->&Self::Output{self.get(p).expect("declared shared attention policy")}}
/// Borrowed iterator; prepared storage never creates a map or copies a tensor.
pub struct Iter<'a,T>(Iteration<'a,T>);
enum Iteration<'a,T>{Ordinary(std::collections::hash_map::Iter<'a,AttentionPolicy,(T,T)>),Prepared(std::slice::Iter<'a,(AttentionPolicy,(T,T))>)}
impl<T> std::fmt::Debug for Iter<'_,T>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.debug_struct("SharedAssistantIter").field("remaining",&self.len()).finish()}}
impl<'a,T> Iterator for Iter<'a,T>{type Item=(&'a AttentionPolicy,&'a(T,T));fn next(&mut self)->Option<Self::Item>{match &mut self.0 {Iteration::Ordinary(v)=>v.next(),Iteration::Prepared(v)=>v.next().map(|(k,v)|(k,v))}}fn size_hint(&self)->(usize,Option<usize>){match &self.0{Iteration::Ordinary(v)=>v.size_hint(),Iteration::Prepared(v)=>v.size_hint()}}}
impl<T> ExactSizeIterator for Iter<'_,T>{}
impl<'a,T> IntoIterator for &'a SharedAssistantStates<T>{type Item=(&'a AttentionPolicy,&'a(T,T));type IntoIter=Iter<'a,T>;fn into_iter(self)->Self::IntoIter{self.iter()}}

impl<T> Clone for Iter<'_,T>{fn clone(&self)->Self{Self(match &self.0{Iteration::Ordinary(v)=>Iteration::Ordinary(v.clone()),Iteration::Prepared(v)=>Iteration::Prepared(v.clone())})}}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc,atomic::{AtomicUsize,Ordering}};
    #[derive(Debug)]
    struct Value{number:u32,dropped:Arc<AtomicUsize>,cloned:Arc<AtomicUsize>}
    impl Clone for Value{fn clone(&self)->Self{self.cloned.fetch_add(1,Ordering::SeqCst);Self{number:self.number,dropped:self.dropped.clone(),cloned:self.cloned.clone()}}}
    impl Drop for Value{fn drop(&mut self){self.dropped.fetch_add(1,Ordering::SeqCst);}}
    struct Host{dropped:Arc<AtomicUsize>,retired:Arc<AtomicUsize>}
    impl Drop for Host{fn drop(&mut self){assert_eq!(self.dropped.load(Ordering::SeqCst),2);self.retired.fetch_add(1,Ordering::SeqCst);}}
    #[test]
    fn prepared_assistant_context_shares_values_and_retires_rows_before_host(){
        let dropped=Arc::new(AtomicUsize::new(0));let cloned=Arc::new(AtomicUsize::new(0));let retired=Arc::new(AtomicUsize::new(0));
        let host=eredu_core::HostPreparationAuthority::retain(Host{dropped:dropped.clone(),retired:retired.clone()});
        let mut rows=eredu_core::SpeculativeBuffer::try_new_retained(1,host.clone()).unwrap();
        rows.try_push((AttentionPolicy::Full,(Value{number:13,dropped:dropped.clone(),cloned:cloned.clone()},Value{number:29,dropped:dropped.clone(),cloned:cloned.clone()}))).unwrap();
        let source=SharedAssistantStates::from_prepared(SpeculativeValues::from_prepared_buffer(rows,host));
        let saved=source.clone();assert_eq!(cloned.load(Ordering::SeqCst),0);
        assert!(std::ptr::eq(source.get(&AttentionPolicy::Full).unwrap(),saved.get(&AttentionPolicy::Full).unwrap()));
        assert_eq!(saved[&AttentionPolicy::Full].0.number,13);assert_eq!(saved[&AttentionPolicy::Full].1.number,29);
        drop(source);assert_eq!(dropped.load(Ordering::SeqCst),0);assert_eq!(retired.load(Ordering::SeqCst),0);
        drop(saved);assert_eq!(dropped.load(Ordering::SeqCst),2);assert_eq!(retired.load(Ordering::SeqCst),1);
    }
    #[test]
    fn ordinary_assistant_context_preserves_policy_publication(){
        let mut source=SharedAssistantStates::from(SharedAttentionStates::from([(AttentionPolicy::Full,(7,11))]));
        source.publish(AttentionPolicy::Full,(17,23)).unwrap();
        assert_eq!(source.values().copied().collect::<Vec<_>>(),[(17,23)]);
        let frozen=SharedAssistantStates::from_prepared(vec![(AttentionPolicy::Full,(17,23))].into());
        assert_eq!(frozen.get(&AttentionPolicy::Full),source.get(&AttentionPolicy::Full));
    }
}
