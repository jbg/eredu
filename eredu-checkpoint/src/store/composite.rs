//! Closed GGUF union construction over already retained immutable catalogs.
use super::storage::SourceControl;
use super::*;
use crate::gguf_store::GgufWeightStore;
use crate::prepared_index::{PreparedIndex, PreparedIndexNode};
use std::{
    alloc::Layout,
    any::Any,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

type Node = PreparedIndexNode<String, usize, ()>;
#[derive(Debug, Default)]
pub(super) struct OwnerRows(PreparedIndex<String, usize, ()>);
impl OwnerRows {
    pub(super) fn get(&self, key: &str) -> Option<&usize> {
        self.0.get_by(|stored| key.cmp(stored.as_str()))
    }
    pub(super) fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(key, _)| key)
    }
    pub(super) fn insert(&mut self, key: String, owner: usize) -> Option<usize> {
        let mut candidate = Some(Node::new(key, owner, ()));
        self.install(&mut candidate)
    }
    fn install(&mut self, candidate: &mut Option<Node>) -> Option<usize> {
        if let Some(previous) = self.get(candidate.as_ref().expect("candidate").key()) {
            return Some(*previous);
        }
        assert!(self.0.insert(candidate), "prechecked immutable union key");
        None
    }
}

#[derive(Debug)]
enum GgufChild {
    Ordinary(Arc<GgufWeightStore>),
    Retained(RetainedCheckpointSource),
}
impl GgufChild {
    fn source(&self) -> &GgufWeightStore {
        match self {
            Self::Ordinary(source) => source,
            Self::Retained(source) => source.gguf().expect("validated typed GGUF child"),
        }
    }
    fn retained(&self) -> Option<&RetainedCheckpointSource> {
        match self {
            Self::Retained(source) => Some(source),
            _ => None,
        }
    }
    fn clone_root(&self) -> RetainedCheckpointSource {
        match self {
            Self::Ordinary(source) => source.clone().into(),
            Self::Retained(source) => source.clone(),
        }
    }
}

/// Both original roots on a closed-pair type refusal. No callbacks or erasure
/// allocations are attempted to turn an arbitrary root into a built-in source.
#[derive(Debug)]
pub struct GgufCompositeInputError([RetainedCheckpointSource; 2]);
impl GgufCompositeInputError {
    /// Actual original pair, retained without extraction or replacement.
    pub fn sources(&self) -> &[RetainedCheckpointSource; 2] {
        &self.0
    }
}
impl fmt::Display for GgufCompositeInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GGUF composite requires two closed retained GGUF sources")
    }
}
impl std::error::Error for GgufCompositeInputError {}

/// Exact built-in pair retained before composing a primary/companion union.
/// Both catalogs are immutable; requesting layouts invokes no source callback.
/// Existing stores and their outer Arc allocations remain separate owners.
#[derive(Debug)]
pub struct GgufCompositePlan {
    sources: [GgufChild; 2],
}
impl GgufCompositePlan {
    /// Retain the original two stores without allocating a plan container.
    pub fn new(first: Arc<GgufWeightStore>, second: Arc<GgufWeightStore>) -> Self {
        Self {
            sources: [GgufChild::Ordinary(first), GgufChild::Ordinary(second)],
        }
    }
    /// Retain the two actual closed GGUF root owners without producing an Arc.
    /// Wrong-kind input preserves the entire original pair in the error.
    pub fn from_retained(
        first: RetainedCheckpointSource,
        second: RetainedCheckpointSource,
    ) -> Result<Self, GgufCompositeInputError> {
        if first.gguf().is_none() || second.gguf().is_none() {
            return Err(GgufCompositeInputError([first, second]));
        }
        Ok(Self {
            sources: [GgufChild::Retained(first), GgufChild::Retained(second)],
        })
    }
    /// Borrow actual input values without erasure, reconstruction or selection.
    pub fn sources(&self) -> [&GgufWeightStore; 2] {
        [self.sources[0].source(), self.sources[1].source()]
    }
    /// Actual retained roots where the closed pair constructor was used.
    /// Ordinary typed Arcs carry no claim about their outer allocation.
    pub fn retained_sources(&self) -> Option<[&RetainedCheckpointSource; 2]> {
        Some([self.sources[0].retained()?, self.sources[1].retained()?])
    }
    /// Fresh child-vector/key/node requests and finite constructor controls.
    /// The runtime qualifies and debits before calling the owning worker.
    pub fn requested_storage<C>(&self) -> Option<GgufCompositeStorageRequest> {
        let mut bytes = Layout::array::<RetainedCheckpointSource>(2).ok()?.size();
        let mut iteration = 0;
        for source in &self.sources {
            let keys = source.source().catalog_keys();
            iteration = iteration.max(size_of_val(&keys));
            for key in keys {
                bytes = bytes
                    .checked_add(key.len())?
                    .checked_add(Node::storage_layout().size())?;
            }
        }
        let controls = [
            size_of::<Self>(),
            size_of::<Construction>(),
            size_of::<GgufCompositeBuildFailure>(),
            size_of::<Cause>(),
            size_of::<CompositeCheckpointSource>(),
            size_of::<Result<CompositeCheckpointSource, GgufCompositeBuildFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Option<Node>>(),
            size_of::<String>(),
            size_of::<&String>(),
            size_of::<Option<&String>>(),
            size_of::<usize>(),
            size_of::<Option<usize>>(),
            size_of::<usize>(), // independent key-iteration request peak
            size_of::<Layout>(),
            size_of::<Option<Layout>>(),
            size_of::<&str>(), // exact get_by comparison capture
            size_of::<Option<&usize>>(),
            size_of::<&mut OwnerRows>(),
            size_of::<&mut Option<Node>>(),
            size_of::<std::cmp::Ordering>(),
            size_of::<bool>(),
            size_of::<std::slice::Iter<'static, GgufChild>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'static, GgufChild>>>(),
            size_of::<std::array::IntoIter<GgufChild, 2>>(),
            size_of::<Arc<GgufWeightStore>>(),
            size_of::<RetainedCheckpointSource>(),
            size_of::<GgufChild>(),
            size_of::<GgufCompositeInputError>(),
            size_of::<Result<GgufCompositePlan, GgufCompositeInputError>>(),
            size_of::<Option<&GgufWeightStore>>(),
            size_of::<[&GgufWeightStore; 2]>(),
            size_of::<Option<[&RetainedCheckpointSource; 2]>>(),
            iteration,
            SourceControl::owner_control_bytes::<C>()?,
            RecipeInferenceCache::initial_control_bytes()?,
            PreparedIndex::<String, usize, ()>::worker_control_layout()?.size(),
        ];
        bytes = controls
            .into_iter()
            .try_fold(bytes, usize::checked_add)?
            .checked_add(size_of_val(&controls))?;
        Some(GgufCompositeStorageRequest {
            bytes,
            custody: SourceControl::body_layout::<C>(),
        })
    }
    /// Use the same owned union worker as supplied construction, without any
    /// accounting claim. This performs no payload read or artifact reopen.
    pub fn build(self) -> Result<CompositeCheckpointSource, GgufCompositeBuildFailure> {
        prepare(self, None)
    }
    /// Execute after the caller has admitted these exact requests. A caller's C
    /// cannot impersonate the runtime's private source-account origin.
    pub fn build_with_custody<C: Any + fmt::Debug + Send + Sync>(
        self,
        custody: C,
    ) -> Result<CompositeCheckpointSource, GgufCompositeBuildFailure> {
        prepare(self, Some(SourceControl::new(custody)))
    }
}

/// Owning-module requests; no capacity or byte grant is accepted from callers.
#[derive(Clone, Copy, Debug)]
pub struct GgufCompositeStorageRequest {
    bytes: usize,
    custody: Layout,
}
impl GgufCompositeStorageRequest {
    /// Fresh Vec, cloned strings, prepared Boxes and named control transports.
    pub fn requested_bytes(&self) -> usize {
        self.bytes
    }
    /// Shared constructor-custody payload, excluding its allocator headers.
    pub fn custody_body(&self) -> Layout {
        self.custody
    }
    /// The two preinitialized empty recipe-cache PAL mutex owners.
    pub fn mutex_count(&self) -> usize {
        2
    }
}

#[derive(Debug)]
enum Cause {
    Reserve(TryReserveError),
    Duplicate { previous: usize, current: usize },
}
struct Construction {
    input: [GgufChild; 2],
    candidate: Option<Node>,
    owners: OwnerRows,
    sources: Vec<RetainedCheckpointSource>,
    recipes: Option<RecipeInferenceCache>,
    // Every actual node, string, vector and private mutex retires first.
    control: Option<SourceControl>,
}
/// Same input pair, duplicate candidate and successful prefix on refusal.
/// No new error strings, descriptor copies or source callbacks are needed.
pub struct GgufCompositeBuildFailure {
    cause: Cause,
    construction: Construction,
}
impl GgufCompositeBuildFailure {
    /// Original stores remain strongly retained by the failed constructor.
    pub fn sources(&self) -> [&GgufWeightStore; 2] {
        [
            self.construction.input[0].source(),
            self.construction.input[1].source(),
        ]
    }
    /// Number of successfully installed rows preceding the failed candidate.
    pub fn completed_rows(&self) -> usize {
        self.construction.owners.0.len()
    }
    /// The original duplicate key buffer; no diagnostic clone is made.
    pub fn duplicate_key(&self) -> Option<&str> {
        matches!(self.cause, Cause::Duplicate { .. }).then(|| {
            self.construction
                .candidate
                .as_ref()
                .expect("duplicate candidate")
                .key()
                .as_str()
        })
    }
}
impl fmt::Debug for GgufCompositeBuildFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GgufCompositeBuildFailure")
            .field("cause", &self.cause)
            .field("completed_rows", &self.completed_rows())
            .field("duplicate_key", &self.duplicate_key())
            .finish()
    }
}
impl fmt::Display for GgufCompositeBuildFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Reserve(cause) => fmt::Display::fmt(cause, f),
            Cause::Duplicate { previous, current } => write!(
                f,
                "composite checkpoint key {:?} is owned by sources {previous} and {current}",
                self.duplicate_key().expect("duplicate key")
            ),
        }
    }
}
impl std::error::Error for GgufCompositeBuildFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Reserve(cause) => Some(cause),
            Cause::Duplicate { .. } => None,
        }
    }
}
fn prepare(
    plan: GgufCompositePlan,
    control: Option<SourceControl>,
) -> Result<CompositeCheckpointSource, GgufCompositeBuildFailure> {
    let mut state = Construction {
        input: plan.sources,
        candidate: None,
        owners: OwnerRows::default(),
        sources: Vec::new(),
        recipes: None,
        control,
    };
    if let Err(cause) = state.build() {
        return Err(GgufCompositeBuildFailure {
            cause,
            construction: state,
        });
    }
    let Construction {
        input,
        candidate,
        owners,
        sources,
        recipes,
        control,
    } = state;
    drop(candidate);
    drop(input);
    Ok(CompositeCheckpointSource {
        recipes: recipes.expect("initialized"),
        sources,
        owners,
        control,
    })
}
impl Construction {
    fn build(&mut self) -> Result<(), Cause> {
        self.sources.try_reserve_exact(2).map_err(Cause::Reserve)?;
        for (owner, source) in self.input.iter().enumerate() {
            for key in source.source().catalog_keys() {
                #[cfg(test)]
                if FAIL_AFTER.with(|n| n.get() == Some(self.owners.0.len())) {
                    return Err(Cause::Reserve(
                        self.sources.try_reserve_exact(usize::MAX).unwrap_err(),
                    ));
                }
                self.candidate = Some(Node::new(key.clone(), owner, ()));
                if let Some(previous) = self.owners.install(&mut self.candidate) {
                    return Err(Cause::Duplicate {
                        previous,
                        current: owner,
                    });
                }
            }
            self.sources.push(source.clone_root());
        }
        self.recipes = Some(RecipeInferenceCache::default());
        self.recipes.as_ref().unwrap().initialize_control_storage();
        Ok(())
    }
}

#[cfg(test)]
thread_local! { static FAIL_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }

#[cfg(test)]
mod tests;
