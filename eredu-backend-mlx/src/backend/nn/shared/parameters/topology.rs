//! One immutable native parameter table for ordinary and funded construction.
use super::*;
use eredu_core::{SharedStorageOwner, SharedStorageRetirement};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError, WorkspaceContext};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::Arc,
};

/// Backend table loans grant no parameter, inventory or graph authority.
pub(in crate::backend::nn::shared) trait NativeParameterTopology {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn get(&self, name: &str) -> Option<&ParameterSpec>;
    fn key(&self, index: usize) -> Option<&str>;
    fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }
}
// Malformed fixture inputs remain borrowed; production has only the row table.
#[cfg(test)]
impl NativeParameterTopology for BTreeMap<String, ParameterSpec> {
    fn len(&self) -> usize {
        BTreeMap::len(self)
    }
    fn get(&self, name: &str) -> Option<&ParameterSpec> {
        BTreeMap::get(self, name)
    }
    fn key(&self, index: usize) -> Option<&str> {
        self.keys().nth(index).map(String::as_str)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("parameter table differs from its exact native fields")]
    Identity,
    #[error("parameter table extent overflowed")]
    Overflow,
    #[error("parameter table source: {0}")]
    Source(#[source] ParameterSourceError),
    #[error("parameter table funding: {0}")]
    Funding(#[source] HostMetadataFundingError),
    #[error("parameter table destination: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    funding: Option<HostMetadataFunding>,
}
fn failure(cause: Cause, funding: Option<&HostMetadataFunding>) -> ComputeError {
    ComputeError::backend_retained_source(Failure {
        cause,
        funding: funding.cloned(),
    })
}
struct Rows {
    rows: Vec<(&'static str, ParameterSpec)>,
    // Every row and the final shared control retire before their account.
    funding: Option<HostMetadataFunding>,
}
impl SharedStorageRetirement for Rows {
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
/// Shared immutable rows; construction policy changes custody, not representation.
#[derive(Debug, Clone)]
pub(in crate::backend::nn::shared) struct NativeParameterTable(SharedStorageOwner<Rows>);
impl NativeParameterTopology for NativeParameterTable {
    fn len(&self) -> usize {
        self.0.rows.len()
    }
    fn get(&self, name: &str) -> Option<&ParameterSpec> {
        self.0
            .rows
            .binary_search_by(|(key, _)| key.cmp(&name))
            .ok()
            .map(|index| &self.0.rows[index].1)
    }
    fn key(&self, index: usize) -> Option<&str> {
        self.0.rows.get(index).map(|(name, _)| *name)
    }
}
pub(in crate::backend::nn::shared) struct TopologyBuilder {
    rows: Vec<(&'static str, ParameterSpec)>,
    limit: usize,
    failed: bool,
    funding: Option<HostMetadataFunding>,
}
impl NativeParameterTable {
    pub(super) fn keys(&self) -> impl ExactSizeIterator<Item = &str> {
        self.0.rows.iter().map(|(name, _)| *name)
    }
    /// Explicit ordinary ownership boundary; no funding or admitted source is inferred.
    pub(in crate::backend::nn::shared) fn from_rows(
        mut rows: Vec<(&'static str, ParameterSpec)>,
    ) -> Result<Self, ComputeError> {
        rows.sort_unstable_by_key(|(name, _)| *name);
        if rows.iter().any(|(name, _)| name.is_empty())
            || rows.windows(2).any(|pair| pair[0].0 == pair[1].0)
        {
            return Err(failure(Cause::Identity, None));
        }
        Ok(Self(SharedStorageOwner::new(Rows {
            rows,
            funding: None,
        })))
    }
    /// Caller-owned specifications have already been copied under their policy.
    pub(in crate::backend::nn::shared) fn control_bytes(rows: usize) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<TopologyBuilder>(),
            size_of::<Rows>(),
            size_of::<SharedStorageOwner<Rows>>(),
            size_of::<Result<Self, ComputeError>>(),
            size_of::<Result<TopologyBuilder, ComputeError>>(),
            size_of::<(usize, &'static str, ParameterSpec)>(),
            size_of::<Option<&HostMetadataFunding>>(),
            Layout::array::<(&'static str, ParameterSpec)>(rows)
                .ok()?
                .size(),
            WorkspaceContext::metadata_arc_bytes::<Rows>()?,
            sources::binding_visit_control_bytes()?,
            ComputeError::retained_source_construction_bytes::<Failure>()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// One exact row reservation, before any push or final shared control.
    pub(in crate::backend::nn::shared) fn prepare(
        rows: usize,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<TopologyBuilder, ComputeError> {
        if let Some(funding) = funding {
            let controls =
                Self::control_bytes(rows).ok_or_else(|| failure(Cause::Overflow, Some(funding)))?;
            funding
                .reserve_metadata(controls)
                .map_err(|cause| failure(Cause::Funding(cause), Some(funding)))?;
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(rows)
            .map_err(|cause| failure(Cause::Allocation(cause), funding))?;
        if values.capacity() > rows {
            return Err(failure(Cause::Overflow, funding));
        }
        Ok(TopologyBuilder {
            rows: values,
            limit: rows,
            failed: false,
            funding: funding.cloned(),
        })
    }
    pub(super) fn validate<M: NativeRetainedValues>(&self, module: &M) -> Result<(), ComputeError> {
        struct Check;
        impl<'a> ParameterSourceVisitor<'a, MlxTensor> for Check {
            fn parameter(&mut self, _: ParameterMetadataView<'a>, _: &'a MlxTensor) {}
            fn retained(&mut self, _: &'a MlxTensor) {}
        }
        visit_module_parameter_sources(module, self, &mut Check)
            .map_err(|cause| failure(Cause::Source(cause), self.0.funding.as_ref()))
    }
    pub(super) fn validate_fields<M: NativeRetainedValues>(
        &self,
        module: &M,
    ) -> Result<(), ComputeError> {
        sources::validate_module_parameter_fields(module, self)
            .map_err(|cause| failure(Cause::Source(cause), self.0.funding.as_ref()))
    }
}
impl TopologyBuilder {
    pub(in crate::backend::nn::shared) fn push(
        &mut self,
        name: &'static str,
        spec: ParameterSpec,
    ) -> Result<(), ComputeError> {
        if self.failed
            || name.is_empty()
            || self.rows.len() == self.limit
            || self.rows.last().is_some_and(|(key, _)| *key >= name)
        {
            self.failed = true;
            return Err(failure(Cause::Identity, self.funding.as_ref()));
        }
        self.rows.push((name, spec));
        Ok(())
    }
    pub(in crate::backend::nn::shared) fn finish(
        self,
    ) -> Result<NativeParameterTable, ComputeError> {
        if self.failed || self.rows.len() != self.limit {
            return Err(failure(Cause::Identity, self.funding.as_ref()));
        }
        Ok(NativeParameterTable(SharedStorageOwner::new(Rows {
            rows: self.rows,
            funding: self.funding,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::HostPreparationAuthority;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct Retained(Arc<AtomicBool>);
    impl Drop for Retained {
        fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); }
    }
    fn funding(bytes: usize) -> (HostMetadataFunding, Arc<AtomicBool>) {
        let retired = Arc::new(AtomicBool::new(false));
        let authority = HostPreparationAuthority::retain(Retained(retired.clone()));
        let controls = HostMetadataFunding::prepaid_control_bytes().unwrap();
        (HostMetadataFunding::from_prepaid(controls + bytes, authority).unwrap(), retired)
    }
    fn spec(name: &str) -> ParameterSpec { ParameterSpec::trainable(name).unwrap() }

    #[test]
    fn ordinary_and_funded_tables_preserve_the_same_names_and_borrowed_specs() {
        let ordinary = NativeParameterTable::from_rows(vec![
            ("weight", spec("projection.weight")), ("bias", spec("projection.bias")),
        ]).unwrap();
        let (funding, retired) = funding(NativeParameterTable::control_bytes(2).unwrap());
        let mut builder = NativeParameterTable::prepare(2, Some(&funding)).unwrap();
        assert_eq!(builder.rows.capacity(), 2);
        builder.push("bias", spec("projection.bias")).unwrap();
        builder.push("weight", spec("projection.weight")).unwrap();
        let paid = builder.finish().unwrap();
        assert_eq!(ordinary.keys().collect::<Vec<_>>(), ["bias", "weight"]);
        for name in ordinary.keys() { assert_eq!(ordinary.get(name), paid.get(name)); }
        let alias = paid.clone();
        assert!(std::ptr::eq(paid.get("weight").unwrap(), alias.get("weight").unwrap()));
        drop((funding, paid));
        assert!(!retired.load(Ordering::SeqCst));
        drop(alias);
        assert!(retired.load(Ordering::SeqCst));
    }

    #[test]
    fn exact_table_refusal_and_failed_row_order_keep_funding_until_failure_retires() {
        let bytes = NativeParameterTable::control_bytes(2).unwrap();
        let (short, retired) = funding(bytes - 1);
        let error = match NativeParameterTable::prepare(2, Some(&short)) {
            Ok(_) => panic!("short account accepted table"), Err(error) => error,
        };
        drop(short);
        assert!(!retired.load(Ordering::SeqCst));
        drop(error);
        assert!(retired.load(Ordering::SeqCst));

        let (funding, retired) = funding(bytes);
        let mut builder = NativeParameterTable::prepare(2, Some(&funding)).unwrap();
        builder.push("weight", spec("weight")).unwrap();
        let error = builder.push("bias", spec("bias")).unwrap_err();
        drop((builder, funding));
        assert!(!retired.load(Ordering::SeqCst));
        drop(error);
        assert!(retired.load(Ordering::SeqCst));
    }
}
