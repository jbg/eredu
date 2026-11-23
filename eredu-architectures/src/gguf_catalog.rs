use std::collections::{hash_map::Entry, BTreeSet, HashMap, HashSet};
use std::hash::BuildHasher;

use eredu_gguf::{Checkpoint, MetadataValue};

/// Supplies the vocabulary size from the canonical embedding matrix when the
/// architecture metadata does not declare it.
pub(crate) fn supply_embedding_vocabulary(
    metadata: &mut HashMap<String, MetadataValue>,
    vocabulary_key: String,
    checkpoint: &Checkpoint,
) {
    if let Entry::Vacant(entry) = metadata.entry(vocabulary_key) {
        if let Some(vocabulary) = checkpoint
            .logical_outputs()
            .find(|output| output.name == "token_embd.weight")
            .and_then(|output| match output.shape.as_slice() {
                [vocabulary, _] => Some(*vocabulary),
                _ => None,
            })
        {
            entry.insert(MetadataValue::Uint64(vocabulary));
        }
    }
}

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
