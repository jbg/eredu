use std::collections::{BTreeSet, HashSet};
use std::hash::BuildHasher;

use eredu_gguf::Checkpoint;

/// Backend-neutral access to the physical tensor names in a GGUF checkpoint.
pub trait GgufTensorCatalog {
    /// Whether one exact physical tensor is present.
    fn contains(&self, name: &str) -> bool;

    /// Whether any physical tensor name satisfies a predicate.
    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool;
}

impl GgufTensorCatalog for Checkpoint {
    fn contains(&self, name: &str) -> bool {
        self.tensors()
            .any(|tensor| tensor.descriptor().name == name)
    }

    fn any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
        self.tensors()
            .any(|tensor| predicate(&tensor.descriptor().name))
    }
}

impl GgufTensorCatalog for BTreeSet<String> {
    fn contains(&self, name: &str) -> bool {
        BTreeSet::contains(self, name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        self.iter().map(String::as_str).any(predicate)
    }
}

impl<S: BuildHasher> GgufTensorCatalog for HashSet<String, S> {
    fn contains(&self, name: &str) -> bool {
        HashSet::contains(self, name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        self.iter().map(String::as_str).any(predicate)
    }
}
