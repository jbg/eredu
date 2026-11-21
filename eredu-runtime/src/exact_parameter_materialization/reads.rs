//! Shared scheduling for exact parameter reads.

use std::collections::{BTreeMap, VecDeque};

use eredu_checkpoint::store::{CheckpointSource, ReadPolicy, StoreError, TensorReadRequest};
use eredu_core::{
    checkpoint_mechanisms::{ParameterReadInput, ParameterReadScope},
    residency::parameter_binding::WeightBinding,
    BackendFailure,
};

use crate::ParameterBatchBudget;

pub(crate) fn ordinary_input<'a>(
    source: &'a dyn CheckpointSource,
    binding: &'a WeightBinding,
) -> Result<ParameterReadInput<'a>, StoreError> {
    if let Some(recipe) = binding.recipe() {
        Ok(ParameterReadInput::Recipe(recipe, source))
    } else {
        source
            .acquire_lease(TensorReadRequest {
                key: binding.checkpoint_key().to_owned(),
                selection: binding.selection().clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .map(ParameterReadInput::Lease)
    }
}

/// Groups adjacent direct reads within the portable byte and parameter bounds.
pub(crate) struct DirectReads<T> {
    budget: ParameterBatchBudget,
    reads: Vec<T>,
}

impl<T> DirectReads<T> {
    pub(crate) fn new() -> Self {
        Self {
            budget: ParameterBatchBudget::default(),
            reads: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, bytes: u64, read: T) -> Option<Vec<T>> {
        let full = if self.budget.try_push(bytes) {
            None
        } else {
            let full = self.take();
            self.budget.try_push(bytes);
            full
        };
        self.reads.push(read);
        full
    }

    pub(crate) fn take(&mut self) -> Option<Vec<T>> {
        self.budget = ParameterBatchBudget::default();
        (!self.reads.is_empty()).then(|| std::mem::take(&mut self.reads))
    }
}

enum Batch<'a, R> {
    Direct(Vec<(&'a WeightBinding, R)>),
    Ordinary(&'a WeightBinding),
}

fn finish<S: ParameterReadScope>(
    scope: &mut S,
    names: Vec<&WeightBinding>,
    pending: S::Pending,
    owners: &mut BTreeMap<String, S::Value>,
) -> Result<(), S::Error> {
    let values = scope.take_values(pending)?;
    insert_values(names.into_iter().map(WeightBinding::name), values, owners)
        .map_err(S::Error::from)
}

fn insert_values<'a, V>(
    names: impl ExactSizeIterator<Item = &'a str>,
    values: Vec<V>,
    owners: &mut BTreeMap<String, V>,
) -> Result<(), BackendFailure> {
    if names.len() != values.len() {
        return Err(super::invalid(
            "native read returned an incorrect number of values",
        ));
    }
    for (name, value) in names.zip(values) {
        if owners.insert(name.to_owned(), value).is_some() {
            return Err(super::invalid(format!(
                "duplicate physical parameter {name:?}"
            )));
        }
    }
    Ok(())
}

/// Materializes exact physical owners and returns them only after all batches succeed.
pub fn materialize_parameter_owners<'a, S: ParameterReadScope>(
    source: &dyn CheckpointSource,
    owners: impl IntoIterator<Item = &'a WeightBinding>,
    scope: &mut S,
) -> Result<BTreeMap<String, S::Value>, S::Error> {
    let classified = classify_reads(source, owners, scope)?;
    materialize_classified_reads(source, classified, scope)
}

pub(crate) type ClassifiedReads<'a, R> = Vec<(&'a WeightBinding, Option<R>)>;

pub(crate) fn classify_reads<'a, S: ParameterReadScope>(
    source: &dyn CheckpointSource,
    owners: impl IntoIterator<Item = &'a WeightBinding>,
    scope: &S,
) -> Result<ClassifiedReads<'a, S::DirectRead>, S::Error> {
    owners
        .into_iter()
        .map(|binding| {
            scope
                .direct_read(source, binding)
                .map(|read| (binding, read))
        })
        .collect()
}

pub(crate) fn materialize_classified_reads<S: ParameterReadScope>(
    source: &dyn CheckpointSource,
    classified: ClassifiedReads<'_, S::DirectRead>,
    scope: &mut S,
) -> Result<BTreeMap<String, S::Value>, S::Error> {
    let mut batches = Vec::new();
    let mut reads = DirectReads::new();
    for (binding, read) in classified {
        if let Some(read) = read {
            if let Some(batch) = reads.push(binding.expected_bytes(), (binding, read)) {
                batches.push(Batch::Direct(batch));
            }
        } else {
            if let Some(batch) = reads.take() {
                batches.push(Batch::Direct(batch));
            }
            batches.push(Batch::Ordinary(binding));
        }
    }
    if let Some(batch) = reads.take() {
        batches.push(Batch::Direct(batch));
    }

    let slots = scope.slots().get();
    let mut pending = VecDeque::new();
    let mut owners = BTreeMap::new();
    for batch in batches {
        let submitted = match batch {
            Batch::Direct(reads) => {
                let (names, reads) = reads
                    .into_iter()
                    .map(|(binding, read)| (binding, (read, binding.storage_layout())))
                    .unzip();
                (names, scope.submit_direct(reads)?)
            }
            Batch::Ordinary(binding) => {
                let mut released_aggregate = false;
                loop {
                    let result = match ordinary_input(source, binding) {
                        Ok(input) => {
                            scope
                                .submit_ordinary(binding.name(), input)
                                .map_err(|error| {
                                    let capacity = S::checkpoint_capacity_exhausted(&error);
                                    (error, capacity)
                                })
                        }
                        Err(error) => {
                            let capacity = matches!(error, StoreError::CapacityExhausted { .. });
                            Err((error.into(), capacity))
                        }
                    };
                    match result {
                        Ok(value) => break (vec![binding], value),
                        Err((error, true)) => {
                            if let Some((names, value)) = pending.pop_front() {
                                finish(scope, names, value, &mut owners)?;
                            } else if !released_aggregate && scope.release_checkpoint_capacity()? {
                                released_aggregate = true;
                            } else {
                                return Err(error);
                            }
                        }
                        Err((error, false)) => return Err(error),
                    }
                }
            }
        };
        pending.push_back(submitted);
        if pending.len() == slots {
            let (names, value) = pending.pop_front().expect("slot was just submitted");
            finish(scope, names, value, &mut owners)?;
        }
    }
    for (names, value) in pending {
        finish(scope, names, value, &mut owners)?;
    }
    Ok(owners)
}
