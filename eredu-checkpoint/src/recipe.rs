//! Backend-neutral derived-weight recipes and shape inference.

use std::{
    any::TypeId,
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex, OnceLock},
};

use crate::store::{CheckpointSource, StoreError, TensorMetadata, TensorSelection, WeightStore};
use crate::StoredDtype;

mod encoded_projection;
pub use encoded_projection::{EncodedRecipeMapping, EncodedRecipeMappingPlan, EncodedRecipeConstruction, EncodedRecipeChildren, EncodedRecipeChildrenPlan};
mod read_catalog;
mod read_keys;
pub use read_keys::{EncodedRecipeKeysBuildError, EncodedRecipeKeysPlan, PreparedEncodedRecipeKeys};
pub use read_catalog::{ReadBatchCatalog, ReadBatchCatalogBuildError, ReadBatchCatalogPlan};
mod finite_inference;
use finite_inference::infer_read_metadata;
mod uncached_catalog;
pub use uncached_catalog::UncachedRecipeCatalog;
pub use finite_inference::{
    infer_recipe_bytes, RecipeInferenceError, RecipeInferenceInput, RecipeInferenceLayout,
    RecipeInferencePlan,
};

/// Metadata-only catalog used to validate a derived-weight recipe.
pub trait RecipeCatalog {
    /// Returns source tensor metadata without reading its payload.
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError>;

    /// Allocation-free retained metadata for finite uncached construction.
    /// The catalog must preserve the same immutable authorized entry throughout
    /// a prepared constructor. Arbitrary owned callbacks remain unqualified.
    fn tensor_metadata_borrowed(&self, _key: &str) -> Option<&TensorMetadata> {
        None
    }

    /// Retained inference for this exact immutable catalog. Mutable catalogs
    /// leave this unset; restricted views must use a separate cache.
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        None
    }
}

/// Inference results owned by one immutable catalog, including failed results.
/// Shared subrecipes and concurrent callers compute each result only once.
#[derive(Debug, Default)]
pub struct RecipeInferenceCache {
    entries: Mutex<RecipeResults<RecipeMetadata, RecipeError>>,
    validations: Mutex<HashMap<TypeId, RecipeResults<(), String>>>,
}

type RecipeResults<T, E> = HashMap<DerivedWeightRecipe, Arc<OnceLock<Result<T, E>>>>;

impl RecipeInferenceCache {
    // Both private table locks are reached before sharing the empty cache. Their
    // two Result/guard transports do not allocate table entries.
    pub(crate) fn initial_control_bytes() -> Option<usize> {
        use std::{
            mem::{size_of, size_of_val},
            sync::{LockResult, MutexGuard},
        };
        type Entries = RecipeResults<RecipeMetadata, RecipeError>;
        type Validations = HashMap<TypeId, RecipeResults<(), String>>;
        let controls = [
            size_of::<Self>(),
            size_of::<&Self>(),
            size_of::<MutexGuard<'static, Entries>>(),
            size_of::<LockResult<MutexGuard<'static, Entries>>>(),
            size_of::<MutexGuard<'static, Validations>>(),
            size_of::<LockResult<MutexGuard<'static, Validations>>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }

    // Eager initialization while the enclosing source has no external alias.
    // The two pinned PAL allocations are then unique, never competing first-lock
    // candidates. Future arbitrary recipe/validator entries remain unbounded.
    pub(crate) fn initialize_control_storage(&self) {
        drop(self.entries.lock().expect("new recipe cache"));
        drop(self.validations.lock().expect("new validation cache"));
    }

    /// Retains a caller's cold, metadata-only validation result for this recipe.
    /// `V` identifies one fixed validator: its answer must depend only on the
    /// recipe and this immutable catalog, never mutable device/runtime facts.
    /// The caller owns validation semantics; this stores only success or text,
    /// without retaining the closure or any backend/native resources.
    pub fn validate<V: 'static>(
        &self,
        recipe: &DerivedWeightRecipe,
        validate: impl FnOnce() -> Result<(), String>,
    ) -> Result<(), String> {
        let cell = {
            let mut validators = self
                .validations
                .lock()
                .map_err(|_| "recipe validation cache is poisoned".to_owned())?;
            let entries = validators.entry(TypeId::of::<V>()).or_default();
            if let Some(cell) = entries.get(recipe) {
                Arc::clone(cell)
            } else {
                let cell = Arc::new(OnceLock::new());
                entries.insert(recipe.clone(), Arc::clone(&cell));
                cell
            }
        };
        cell.get_or_init(validate).clone()
    }

    fn with_inferred<C: RecipeCatalog + ?Sized, T>(
        &self,
        recipe: &DerivedWeightRecipe,
        catalog: &C,
        inspect: impl FnOnce(&RecipeMetadata) -> T,
    ) -> Result<T, RecipeError> {
        let cell = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| StoreError::Internal("recipe inference cache is poisoned".into()))?;
            if let Some(cell) = entries.get(recipe) {
                Arc::clone(cell)
            } else {
                let cell = Arc::new(OnceLock::new());
                entries.insert(recipe.clone(), Arc::clone(&cell));
                cell
            }
        };
        // The cache lock ends before initialization and before the caller's
        // inspection. Reentrant inspections therefore never hold this mutex.
        match cell.get_or_init(|| recipe.infer_uncached(catalog)) {
            Ok(metadata) => Ok(inspect(metadata)),
            Err(error) => Err(error.clone()),
        }
    }
}

/// Cold-path capability for proving that every recipe source can be read with
/// its declared physical bound.
pub trait BoundedRecipeSource: RecipeCatalog {
    /// Acquires and immediately releases one source under bounded-read policy.
    fn verify_bounded_source(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<(), StoreError>;
}

impl<T: WeightStore> BoundedRecipeSource for T {
    fn verify_bounded_source(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<(), StoreError> {
        drop(self.acquire(crate::store::TensorReadRequest {
            key: key.to_owned(),
            selection,
            policy: crate::store::ReadPolicy::RequireBounded,
        })?);
        Ok(())
    }
}

impl<T: WeightStore> RecipeCatalog for T {
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        WeightStore::recipe_cache(self)
    }

    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.metadata(key)
    }
}

impl RecipeCatalog for dyn CheckpointSource + '_ {
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        CheckpointSource::recipe_cache(self)
    }

    fn tensor_metadata_borrowed(&self, key: &str) -> Option<&TensorMetadata> {
        self.source_metadata_borrowed(key).ok()
    }

    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source_metadata(key)
    }
}

impl BoundedRecipeSource for dyn CheckpointSource + '_ {
    fn verify_bounded_source(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<(), StoreError> {
        drop(self.acquire_lease(crate::store::TensorReadRequest {
            key: key.to_owned(),
            selection,
            policy: crate::store::ReadPolicy::RequireBounded,
        })?);
        Ok(())
    }
}

/// Scalar representation produced by a recipe operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum RecipeDtype {
    Bool,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
    C64,
    F8E4M3,
    F8E5M2,
    F4,
    F8E8M0,
    Other(String),
}

impl RecipeDtype {
    /// Returns the exact scalar representation width in bits.
    pub fn bit_width(&self) -> Result<u64, RecipeError> {
        match self {
            Self::F4 => Ok(4),
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 => Ok(8),
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => Ok(16),
            Self::I32 | Self::U32 | Self::F32 => Ok(32),
            Self::F64 | Self::I64 | Self::U64 | Self::C64 => Ok(64),
            Self::Other(dtype) => Err(RecipeError::UnsupportedDtype {
                dtype: dtype.clone(),
            }),
        }
    }
}

impl From<StoredDtype> for RecipeDtype {
    fn from(value: StoredDtype) -> Self {
        match value {
            StoredDtype::Bool => Self::Bool,
            StoredDtype::U8 => Self::U8,
            StoredDtype::I8 => Self::I8,
            StoredDtype::I16 => Self::I16,
            StoredDtype::U16 => Self::U16,
            StoredDtype::F16 => Self::F16,
            StoredDtype::BF16 => Self::BF16,
            StoredDtype::I32 => Self::I32,
            StoredDtype::U32 => Self::U32,
            StoredDtype::F32 => Self::F32,
            StoredDtype::F64 => Self::F64,
            StoredDtype::I64 => Self::I64,
            StoredDtype::U64 => Self::U64,
            StoredDtype::C64 => Self::C64,
            StoredDtype::F8E4M3 => Self::F8E4M3,
            StoredDtype::F8E5M2 => Self::F8E5M2,
            StoredDtype::F4 => Self::F4,
            StoredDtype::F8E8M0 => Self::F8E8M0,
            StoredDtype::Other(dtype) => Self::Other(dtype),
        }
    }
}

/// Shape, representation, and byte size inferred for a recipe.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecipeMetadata {
    /// Inferred logical output shape.
    pub shape: Vec<usize>,
    /// Inferred output scalar representation.
    pub dtype: RecipeDtype,
    /// Exact encoded or materialized output byte count.
    pub byte_len: u64,
}

/// Borrowed validated output metadata. Dimensions may apply one source
/// selection replacement without allocating an owned shape. The loan never
/// outlives the metadata/selection supplied to its inspection callback.
#[derive(Clone, Copy, Debug)]
pub struct RecipeMetadataView<'a> {
    shape: SelectedRecipeShape<'a>,
    dtype: &'a RecipeDtype,
    byte_len: u64,
}

#[derive(Clone, Copy, Debug)]
struct SelectedRecipeShape<'a> {
    dimensions: &'a [usize],
    replacement: Option<(usize, usize)>,
}

impl<'a> RecipeMetadataView<'a> {
    /// Exact dimensions in logical axis order.
    pub fn shape(self) -> impl ExactSizeIterator<Item = usize> + Clone + 'a {
        self.shape.iter()
    }
    /// Borrowed scalar representation.
    pub const fn dtype(self) -> &'a RecipeDtype {
        self.dtype
    }
    /// Exact inferred encoded or materialized byte count.
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }
}

/// A validated recipe whose output can be filled directly from encoded ranges.
/// This owns metadata and read admission, never a tensor allocation.
/// Borrowed reads reuse the immutable source records and validate their source
/// on every read. Caller custody retires after output metadata and read records.
/// Cloning is available only when custody itself permits it; cloning metadata
/// has separate storage costs.
#[derive(Clone)]
pub struct EncodedRecipeRead<C = ()> {
    output: RecipeMetadata,
    batch: crate::store::EncodedReadBatch,
    _custody: C,
}

impl<C> EncodedRecipeRead<C> {
    /// Assemble already constructed output metadata and read records without
    /// copying either. A byte-length mismatch retains both owners and custody.
    pub fn from_prepared(
        output: RecipeMetadata, read: crate::store::PreparedEncodedRead<C>,
    ) -> Result<Self, EncodedRecipeReadAssemblyError<C>> {
        if u64::try_from(read.byte_len()).ok() != Some(output.byte_len()) {
            return Err(EncodedRecipeReadAssemblyError { output, read });
        }
        Ok(Self { output, batch: read.batch, _custody: read._custody })
    }

    pub(crate) fn admitted_batch(&self) -> &crate::store::EncodedReadBatch {
        &self.batch
    }

    /// Exact logical shape, dtype, and byte length of the output.
    pub fn output(&self) -> &RecipeMetadata {
        &self.output
    }

    /// Exact source encodings in destination order.
    pub fn sources(&self) -> &[TensorMetadata] {
        self.batch.tensors()
    }

    /// Finite synchronous-read scratch over these exact immutable read plans.
    pub fn borrowed_read_layout<'a>(
        reads: impl IntoIterator<Item = &'a Self>,
    ) -> Option<crate::store::EncodedReadLayout> where C: 'a {
        crate::store::EncodedReadLayout::inspect(reads.into_iter().map(|read| &read.batch))
    }

    /// Fills caller-owned outputs without cloning source metadata or read spans.
    /// The caller retains the read plans and its scratch/error account through
    /// this synchronous operation and discards every output on failure.
    pub fn read_many_borrowed_into<'a, I>(
        reads: I,
        outputs: &mut [&mut [u8]],
    ) -> Result<(), crate::store::EncodedReadFailure>
    where
        I: Iterator<Item = &'a Self> + Clone + ExactSizeIterator,
        C: 'a,
    {
        crate::store::EncodedReadBatch::read_many_borrowed_into(
            reads.map(|read| &read.batch),
            outputs,
        )
    }

    /// Fills the caller's final output allocation in recipe order.
    pub fn read_into(self, output: &mut [u8]) -> Result<(), StoreError> {
        self.batch.read_into(output)
    }
}

impl EncodedRecipeRead {
    /// Plans a detached finite source from these exact admitted recipe reads.
    /// Construction copies metadata only, after the caller supplies source custody.
    pub fn prepare_detached<'a, I>(reads: I) -> Option<crate::store::DetachedEncodedReadPlan<'a, I>>
    where
        I: Iterator<Item = &'a Self> + Clone + ExactSizeIterator,
    {
        crate::store::DetachedEncodedReadPlan::inspect(reads)
    }

    /// Requested backing for one immutable metadata clone. The inline value,
    /// shared source custody and caller-owned destination are separate.
    pub fn clone_storage_bytes(&self) -> Option<usize> {
        let dtype = match &self.output.dtype {
            RecipeDtype::Other(name) => name.len(),
            _ => 0,
        };
        self.batch
            .clone_storage_bytes()?
            .checked_add(
                std::alloc::Layout::array::<usize>(self.output.shape.len())
                    .ok()?
                    .size(),
            )?
            .checked_add(dtype)
    }

    /// Named clone transports, separate from the actual backing requests.
    pub fn clone_control_bytes() -> Option<usize> {
        crate::store::EncodedReadBatch::clone_control_bytes()?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(std::mem::size_of::<RecipeMetadata>())?
            .checked_add(std::mem::size_of::<Vec<usize>>())?
            .checked_add(std::mem::size_of::<String>())
    }

    /// Fills multiple final recipe outputs with a shared shard-ordered read pass.
    pub fn read_many_into(reads: Vec<Self>, outputs: &mut [&mut [u8]]) -> Result<(), StoreError> {
        crate::store::EncodedReadBatch::read_many_into(
            reads.into_iter().map(|read| read.batch).collect(),
            outputs,
        )
    }

}

/// Byte-length disagreement retains actual output metadata and the read owner.
#[derive(Debug)]
pub struct EncodedRecipeReadAssemblyError<C> {
    output: RecipeMetadata,
    read: crate::store::PreparedEncodedRead<C>,
}
impl<C> std::fmt::Display for EncodedRecipeReadAssemblyError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "recipe output has {} bytes but its read has {}", self.output.byte_len(), self.read.byte_len())
    }
}
impl<C: std::fmt::Debug> std::error::Error for EncodedRecipeReadAssemblyError<C> {}

/// A named recipe collection that becomes observable only after every output
/// has passed metadata inference. This is the atomic unit used for fused
/// weights and their affine or FP8 companions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AtomicRecipeSet {
    outputs: BTreeMap<String, DerivedWeightRecipe>,
    aliases: BTreeMap<String, String>,
}

impl AtomicRecipeSet {
    /// Validates all target names, rejects collisions, and infers every recipe
    /// before returning any bindable output.
    pub fn new<C: RecipeCatalog + ?Sized>(
        catalog: &C,
        outputs: impl IntoIterator<Item = (String, DerivedWeightRecipe)>,
    ) -> Result<Self, RecipeError> {
        Self::new_with_aliases(catalog, outputs, std::iter::empty())
    }

    /// Validates canonical outputs and logical aliases as one publication.
    ///
    /// Alias destinations may name another alias in the input declarations,
    /// but the returned map always points directly at a canonical output.
    /// No output or alias is observable when any recipe or alias is invalid.
    pub fn new_with_aliases<C: RecipeCatalog + ?Sized>(
        catalog: &C,
        outputs: impl IntoIterator<Item = (String, DerivedWeightRecipe)>,
        aliases: impl IntoIterator<Item = RecipeAlias>,
    ) -> Result<Self, RecipeError> {
        let mut validated = BTreeMap::new();
        for (target, recipe) in outputs {
            if target.trim().is_empty() {
                return Err(RecipeError::EmptyOutputName);
            }
            if validated.insert(target.clone(), recipe).is_some() {
                return Err(RecipeError::DuplicateOutput { target });
            }
        }
        if validated.is_empty() {
            return Err(RecipeError::EmptyOutputs);
        }
        for recipe in validated.values() {
            recipe.infer(catalog)?;
        }
        let aliases = validate_recipe_aliases(validated.keys(), aliases)?;
        Ok(Self {
            outputs: validated,
            aliases,
        })
    }

    /// Returns the validated recipe for one canonical output.
    pub fn get(&self, target: &str) -> Option<&DerivedWeightRecipe> {
        self.outputs.get(target)
    }

    /// Resolves a canonical output or logical alias without cloning its recipe.
    pub fn get_resolved(&self, target: &str) -> Option<(&str, &DerivedWeightRecipe)> {
        let owner = self
            .aliases
            .get(target)
            .map(String::as_str)
            .unwrap_or(target);
        self.outputs
            .get_key_value(owner)
            .map(|(owner, recipe)| (owner.as_str(), recipe))
    }

    /// Iterates canonical outputs in stable sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &DerivedWeightRecipe)> {
        self.outputs
            .iter()
            .map(|(target, recipe)| (target.as_str(), recipe))
    }

    /// Iterates logical alias and canonical-owner identities in stable order.
    pub fn aliases(&self) -> impl Iterator<Item = (&str, &str)> {
        self.aliases
            .iter()
            .map(|(alias, owner)| (alias.as_str(), owner.as_str()))
    }

    /// Consumes the validated set for a backend binding plan.
    pub fn into_outputs(self) -> BTreeMap<String, DerivedWeightRecipe> {
        self.outputs
    }

    /// Consumes the publication into canonical recipes and logical aliases.
    pub fn into_parts(
        self,
    ) -> (
        BTreeMap<String, DerivedWeightRecipe>,
        BTreeMap<String, String>,
    ) {
        (self.outputs, self.aliases)
    }
}

/// One logical parameter alias published alongside canonical recipes.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecipeAlias {
    /// Logical alias identity.
    pub alias: String,
    /// Canonical output or another declared alias.
    pub destination: String,
}

impl RecipeAlias {
    /// Creates an alias declaration validated when its recipe set is published.
    pub fn new(alias: impl Into<String>, destination: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
            destination: destination.into(),
        }
    }
}

fn validate_recipe_aliases<'a>(
    outputs: impl IntoIterator<Item = &'a String>,
    aliases: impl IntoIterator<Item = RecipeAlias>,
) -> Result<BTreeMap<String, String>, RecipeError> {
    let outputs = outputs.into_iter().cloned().collect::<BTreeSet<_>>();
    let mut declarations = BTreeMap::new();
    for declaration in aliases {
        if declaration.alias.trim().is_empty() {
            return Err(RecipeError::EmptyAliasName);
        }
        if declaration.destination.trim().is_empty() {
            return Err(RecipeError::InvalidAliasDestination {
                alias: declaration.alias,
                destination: declaration.destination,
            });
        }
        if outputs.contains(&declaration.alias) {
            return Err(RecipeError::AliasOutputCollision {
                alias: declaration.alias,
            });
        }
        if declarations
            .insert(declaration.alias.clone(), declaration.destination)
            .is_some()
        {
            return Err(RecipeError::DuplicateAlias {
                alias: declaration.alias,
            });
        }
    }

    fn resolve(
        alias: &str,
        outputs: &BTreeSet<String>,
        declarations: &BTreeMap<String, String>,
        resolved: &mut BTreeMap<String, String>,
        visiting: &mut BTreeSet<String>,
    ) -> Result<String, RecipeError> {
        if let Some(owner) = resolved.get(alias) {
            return Ok(owner.clone());
        }
        if !visiting.insert(alias.to_owned()) {
            return Err(RecipeError::AliasCycle {
                alias: alias.to_owned(),
            });
        }
        let destination = declarations
            .get(alias)
            .expect("resolver receives a declared alias");
        let owner = if outputs.contains(destination) {
            destination.clone()
        } else if declarations.contains_key(destination) {
            resolve(destination, outputs, declarations, resolved, visiting)?
        } else {
            return Err(RecipeError::InvalidAliasDestination {
                alias: alias.to_owned(),
                destination: destination.clone(),
            });
        };
        visiting.remove(alias);
        resolved.insert(alias.to_owned(), owner.clone());
        Ok(owner)
    }

    let mut resolved = BTreeMap::new();
    for alias in declarations.keys() {
        resolve(
            alias,
            &outputs,
            &declarations,
            &mut resolved,
            &mut BTreeSet::new(),
        )?;
    }
    Ok(resolved)
}

/// One named weight or quantization-companion recipe in a matrix family.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MatrixRecipeMember {
    /// Canonical target identity published by the family.
    pub target: String,
    /// Recipe producing the target value.
    pub recipe: DerivedWeightRecipe,
}

impl MatrixRecipeMember {
    /// Creates one member validated when its family is constructed.
    pub fn new(target: impl Into<String>, recipe: DerivedWeightRecipe) -> Self {
        Self {
            target: target.into(),
            recipe,
        }
    }
}

/// Atomic weight, scale, and optional affine-bias recipe family.
///
/// Every companion must have the weight's rank and leading matrix geometry;
/// only the final packed/group dimension may differ. Transformations return a
/// new validated family, so malformed members never become partially visible.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AtomicMatrixRecipeFamily {
    weight: MatrixRecipeMember,
    scales: Option<MatrixRecipeMember>,
    biases: Option<MatrixRecipeMember>,
}

impl AtomicMatrixRecipeFamily {
    /// Validates a complete dense or packed matrix family.
    pub fn new<C: RecipeCatalog + ?Sized>(
        catalog: &C,
        weight: MatrixRecipeMember,
        scales: Option<MatrixRecipeMember>,
        biases: Option<MatrixRecipeMember>,
    ) -> Result<Self, RecipeError> {
        if biases.is_some() && scales.is_none() {
            return Err(RecipeError::MatrixBiasWithoutScales);
        }
        let family = Self {
            weight,
            scales,
            biases,
        };
        family.validate(catalog)?;
        Ok(family)
    }

    /// Canonical weight member.
    pub const fn weight(&self) -> &MatrixRecipeMember {
        &self.weight
    }

    /// Optional quantization-scale member.
    pub const fn scales(&self) -> Option<&MatrixRecipeMember> {
        self.scales.as_ref()
    }

    /// Optional affine-bias member.
    pub const fn biases(&self) -> Option<&MatrixRecipeMember> {
        self.biases.as_ref()
    }

    /// Applies one bounded range or ordered-index selection on leading axis 0.
    ///
    /// The identical logical selection is pushed through the weight and every
    /// present companion before the resulting family is validated atomically.
    pub fn select_leading_axis<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
        selection: TensorSelection,
    ) -> Result<Self, RecipeError> {
        match &selection {
            TensorSelection::Full
            | TensorSelection::Range { axis: 0, .. }
            | TensorSelection::Indices { axis: 0, .. } => {}
            TensorSelection::Range { axis, .. } | TensorSelection::Indices { axis, .. } => {
                return Err(RecipeError::MatrixFamilySelectionAxis { axis: *axis });
            }
            TensorSelection::Contiguous { .. } => {
                return Err(RecipeError::MatrixFamilyContiguousSelection);
            }
        }
        let select = |member: &MatrixRecipeMember| -> Result<MatrixRecipeMember, RecipeError> {
            Ok(MatrixRecipeMember {
                target: member.target.clone(),
                recipe: member.recipe.select_bounded(catalog, selection.clone())?,
            })
        };
        Self::new(
            catalog,
            select(&self.weight)?,
            self.scales.as_ref().map(select).transpose()?,
            self.biases.as_ref().map(select).transpose()?,
        )
    }

    /// Publishes this family and its aliases as one atomic recipe set.
    pub fn publish<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
        aliases: impl IntoIterator<Item = RecipeAlias>,
    ) -> Result<AtomicRecipeSet, RecipeError> {
        self.validate(catalog)?;
        AtomicRecipeSet::new_with_aliases(
            catalog,
            std::iter::once(&self.weight)
                .chain(self.scales.iter())
                .chain(self.biases.iter())
                .map(|member| (member.target.clone(), member.recipe.clone())),
            aliases,
        )
    }

    fn validate<C: RecipeCatalog + ?Sized>(&self, catalog: &C) -> Result<(), RecipeError> {
        let members = std::iter::once(&self.weight)
            .chain(self.scales.iter())
            .chain(self.biases.iter())
            .collect::<Vec<_>>();
        let mut targets = BTreeSet::new();
        for member in &members {
            if member.target.trim().is_empty() {
                return Err(RecipeError::EmptyOutputName);
            }
            if !targets.insert(member.target.clone()) {
                return Err(RecipeError::DuplicateOutput {
                    target: member.target.clone(),
                });
            }
        }
        let weight = self.weight.recipe.infer(catalog)?;
        if weight.shape.len() < 2 {
            return Err(RecipeError::InvalidMatrixFamilyWeight {
                shape: weight.shape,
            });
        }
        let leading = &weight.shape[..weight.shape.len() - 1];
        let mut scale_shape = None;
        for (kind, member) in [
            ("scales", self.scales.as_ref()),
            ("biases", self.biases.as_ref()),
        ] {
            let Some(member) = member else { continue };
            let metadata = member.recipe.infer(catalog)?;
            if metadata.shape.len() != weight.shape.len()
                || metadata.shape[..metadata.shape.len() - 1] != *leading
            {
                return Err(RecipeError::MatrixCompanionGeometry {
                    member: kind,
                    weight: weight.shape.clone(),
                    companion: metadata.shape,
                });
            }
            if kind == "scales" {
                scale_shape = Some(metadata.shape);
            } else if scale_shape.as_ref() != Some(&metadata.shape) {
                return Err(RecipeError::MatrixScaleBiasGeometry {
                    scales: scale_shape.unwrap_or_default(),
                    biases: metadata.shape,
                });
            }
        }
        Ok(())
    }
}

/// One canonical output range cut from a fused source axis.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusedSplitOutput {
    /// Canonical target parameter identity.
    pub target: String,
    /// Positive width along the split axis.
    pub width: usize,
}

/// One physical fused tensor participating in an atomic split family.
///
/// Weight, bias, affine companions, and inverse scales are represented by
/// separate members so each may declare its actual physical axis and widths.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusedSplitMember {
    /// Physical source tensor identity.
    pub source: String,
    /// Axis partitioned into ordered output ranges.
    pub axis: usize,
    /// Canonical targets in physical row order.
    pub outputs: Vec<FusedSplitOutput>,
}

/// Builds an atomic grouped split for one or more fused physical members.
pub fn atomic_fused_split_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    members: impl IntoIterator<Item = FusedSplitMember>,
) -> Result<AtomicRecipeSet, RecipeError> {
    let mut recipes = Vec::new();
    for member in members {
        if member.source.trim().is_empty() {
            return Err(RecipeError::EmptySourceKey);
        }
        if member.outputs.is_empty()
            || member
                .outputs
                .iter()
                .any(|output| output.target.trim().is_empty() || output.width == 0)
        {
            return Err(RecipeError::InvalidFusedSplit {
                tensor: member.source,
            });
        }
        let metadata = catalog.tensor_metadata(&member.source)?;
        let dimension = metadata.logical_shape.get(member.axis).copied().ok_or(
            RecipeError::InvalidSelectionAxis {
                axis: member.axis,
                rank: metadata.logical_shape.len(),
            },
        )?;
        let total = member.outputs.iter().try_fold(0usize, |total, output| {
            total
                .checked_add(output.width)
                .ok_or(RecipeError::ArithmeticOverflow("fused split widths"))
        })?;
        if total != dimension {
            return Err(RecipeError::FusedSplitWidthMismatch {
                tensor: member.source,
                axis: member.axis,
                dimension,
                outputs: total,
            });
        }
        let mut start = 0usize;
        for output in member.outputs {
            let end = start + output.width;
            recipes.push((
                output.target,
                DerivedWeightRecipe::source(
                    &member.source,
                    TensorSelection::Range {
                        axis: member.axis,
                        start,
                        end,
                    },
                ),
            ));
            start = end;
        }
    }
    AtomicRecipeSet::new(catalog, recipes)
}

/// Creates and validates an ordered gather/permutation along one source axis.
/// Duplicate indices are intentionally admitted for broadcast-style layouts;
/// callers that require a permutation must supply unique indices.
pub fn ordered_axis_selection<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    source: impl Into<String>,
    axis: usize,
    indices: Vec<usize>,
) -> Result<DerivedWeightRecipe, RecipeError> {
    let recipe = DerivedWeightRecipe::source(source, TensorSelection::Indices { axis, indices });
    recipe.infer(catalog)?;
    Ok(recipe)
}

impl RecipeMetadata {
    /// Lends the already validated output without cloning its owned fields.
    pub fn borrowed(&self) -> RecipeMetadataView<'_> {
        RecipeMetadataView {
            shape: SelectedRecipeShape {
                dimensions: &self.shape,
                replacement: None,
            },
            dtype: &self.dtype,
            byte_len: self.byte_len,
        }
    }

    /// Returns the inferred output shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the inferred scalar representation.
    pub const fn dtype(&self) -> &RecipeDtype {
        &self.dtype
    }

    /// Returns the exact inferred byte count.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// Typed operations needed to derive a runtime parameter from checkpoint tensors.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[allow(missing_docs)]
pub enum DerivedWeightRecipe {
    Source {
        key: String,
        selection: TensorSelection,
    },
    Select {
        input: Box<Self>,
        selection: TensorSelection,
    },
    Concatenate {
        axis: usize,
        inputs: Vec<Self>,
    },
    Stack {
        axis: usize,
        inputs: Vec<Self>,
    },
    Reshape {
        input: Box<Self>,
        shape: Vec<usize>,
    },
    Transpose {
        input: Box<Self>,
        axes: Vec<usize>,
    },
    Cast {
        input: Box<Self>,
        dtype: RecipeDtype,
    },
    View {
        input: Box<Self>,
        dtype: RecipeDtype,
        shape: Vec<usize>,
    },
    NegLog {
        input: Box<Self>,
    },
    SubtractOne {
        input: Box<Self>,
    },
}

mod input_size;

/// Borrowed visits of actual recipe source occurrences, preserving order and
/// repetitions. Join events describe topology; callers own their storage policy.
/// No source authorization, acquisition, backend behavior or fit is implied.
pub trait RecipeSourceVisitor {
    /// The visitor's own refusal.
    type Error;
    /// One actual declared source selection.
    fn source(&mut self, key: &str, selection: &TensorSelection) -> Result<(), Self::Error>;
    /// Enter an actual concatenate/stack before visiting its children.
    fn enter_join(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Leave a successfully visited concatenate/stack.
    fn leave_join(&mut self) {}
}

trait RecipeSourceVisitorLoan<'a> {
    type Error;
    fn source(&mut self, key: &'a str, selection: &'a TensorSelection) -> Result<(), Self::Error>;
    fn enter_join(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn leave_join(&mut self) {}
}

impl DerivedWeightRecipe {
    /// Compiles encoded-byte recipes into one bounded read batch: joins on any
    /// byte-aligned axis, bounded source selections, reshapes/views, same-dtype
    /// casts, and permutations that move only singleton axes. Reordered rows
    /// map admitted file ranges directly into the final destination; they do
    /// not allocate intermediate payloads or convert scalar representations.
    ///
    /// Unsupported transformations or sources return `None` without payload
    /// reads. Supported recipes are inferred once against the admitted batch's
    /// in-memory metadata, so large expert joins do not re-open every source.
    pub fn prepare_encoded_read(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<Option<EncodedRecipeRead>, RecipeError> {
        self.prepare_encoded_read_with_cache(source, true)
    }

    /// Compiles the same encoded ranges without consulting or growing the
    /// source's persistent recipe-inference cache. Inference uses the exact
    /// metadata retained by this read batch; temporary inference results retire
    /// before return. The caller still owns admission for construction metadata,
    /// the returned read and its later read scratch.
    ///
    /// This preserves the ordinary compiler's geometry, source validation and
    /// unsupported-transform behavior. It does not read tensor payloads.
    pub fn prepare_encoded_read_uncached(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<Option<EncodedRecipeRead>, RecipeError> {
        self.prepare_encoded_read_with_cache(source, false)
    }

    fn prepare_encoded_read_with_cache(
        &self,
        source: &dyn CheckpointSource,
        use_source_cache: bool,
    ) -> Result<Option<EncodedRecipeRead>, RecipeError> {
        // The source batch is already admitted. Use that same catalog for the
        // geometry proof; no source read, payload transform or backend branch.
        fn preserves_bytes<C: RecipeCatalog + ?Sized>(
            recipe: &DerivedWeightRecipe,
            catalog: &C,
        ) -> Result<bool, RecipeError> {
            match recipe {
                DerivedWeightRecipe::Source { .. } => Ok(true),
                DerivedWeightRecipe::Concatenate { inputs, .. }
                | DerivedWeightRecipe::Stack { inputs, .. } => {
                    for input in inputs {
                        if !preserves_bytes(input, catalog)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                DerivedWeightRecipe::Reshape { input, .. }
                | DerivedWeightRecipe::View { input, .. } => preserves_bytes(input, catalog),
                DerivedWeightRecipe::Cast { input, dtype } => {
                    Ok(preserves_bytes(input, catalog)?
                        && infer_read_metadata(input, catalog)?.dtype == *dtype)
                }
                DerivedWeightRecipe::Transpose { input, axes } => {
                    if !preserves_bytes(input, catalog)? {
                        return Ok(false);
                    }
                    let input = infer_read_metadata(input, catalog)?;
                    validate_permutation(axes, input.shape.len())?;
                    // Empty tensors contain no ordered scalar bytes. Otherwise
                    // non-singleton axes must retain their relative order; size
                    // equality alone cannot justify exchanging two real axes.
                    Ok(input.shape.contains(&0)
                        || axes
                            .iter()
                            .copied()
                            .filter(|axis| input.shape[*axis] > 1)
                            .eq((0..input.shape.len()).filter(|axis| input.shape[*axis] > 1)))
                }
                _ => Ok(false),
            }
        }
        let Some(plan) = EncodedRecipeKeysPlan::new(self)? else {
            return Ok(None);
        };
        let contiguous = plan.contiguous();
        let keys = plan.construct(()).map_err(|cause| {
            StoreError::Internal(format!("encoded recipe source keys: {cause}"))
        })?;
        if !contiguous {
            return encoded_projection::prepare(self, source, keys.keys(), use_source_cache);
        }
        let Some(batch) = source.prepare_encoded_read(keys.keys())? else {
            return Ok(None);
        };
        let output = if use_source_cache && source.recipe_cache().is_some() {
            // Immutable sources bind read batches to the same admitted catalog.
            // Validate the whole recipe first, preserving ordinary left-to-right
            // error precedence before checking the byte-preserving subset.
            let output = self.infer(source)?;
            if !preserves_bytes(self, source)? {
                return Ok(None);
            }
            output
        } else {
            let catalog = ReadBatchCatalogPlan::new(batch.tensors())?.construct(())?;
            let output = infer_read_metadata(self, &catalog)?;
            if !preserves_bytes(self, &catalog)? {
                return Ok(None);
            }
            output
        };
        if output.byte_len != batch.byte_len() as u64 {
            // Packed sub-byte tensors may have padding that cannot be joined
            // as bytes. Leave those transformations to the ordinary path.
            return Ok(None);
        }
        Ok(Some(EncodedRecipeRead { output, batch, _custody: () }))
    }

    /// Creates a recipe reading one selected checkpoint tensor.
    pub fn source(key: impl Into<String>, selection: TensorSelection) -> Self {
        Self::Source {
            key: key.into(),
            selection,
        }
    }

    /// Proves that every physical source can honor its declared bounded read.
    pub fn preflight_bounded<S: BoundedRecipeSource + ?Sized>(
        &self,
        source: &S,
    ) -> Result<(), RecipeError> {
        match self {
            Self::Source { key, selection } => {
                source.verify_bounded_source(key, selection.clone())?;
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                for input in inputs {
                    input.preflight_bounded(source)?;
                }
            }
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => input.preflight_bounded(source)?,
        }
        Ok(())
    }

    /// Rewrites an output selection toward physically bounded sources.
    pub fn select_bounded<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
        selection: TensorSelection,
    ) -> Result<Self, RecipeError> {
        let metadata = self.infer(catalog)?;
        let selection = normalize_selection(selection, &metadata.shape)?;
        let expected_shape = selected_shape(metadata.shape.clone(), &selection)?;
        let expanded = expand_indexed_sources(self.clone());
        let rewritten = normalize_bounded_source_ranges(
            expand_indexed_sources(push_selection(&expanded, catalog, selection)?),
            catalog,
        )?;
        let actual = rewritten.infer(catalog)?;
        if actual.shape != expected_shape || actual.dtype != metadata.dtype {
            return Err(RecipeError::SelectionPushdownUnsupported {
                operation: "recipe",
                reason: format!(
                    "rewrite produced shape {:?} and dtype {:?}, expected {:?} and {:?}",
                    actual.shape, actual.dtype, expected_shape, metadata.dtype
                ),
            });
        }
        Ok(rewritten)
    }

    /// Selects every leading-axis member, retaining its singleton axis.
    ///
    /// A stack is validated once and each child is visited only for its own
    /// selection. Repeatedly selecting individual members of a large stack
    /// would otherwise re-infer and clone the entire bank for every member.
    pub fn select_bounded_members<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
    ) -> Result<Vec<Self>, RecipeError> {
        let metadata = self.infer(catalog)?;
        let count = metadata.shape.first().copied().ok_or_else(|| {
            RecipeError::SelectionPushdownUnsupported {
                operation: "member selection",
                reason: "a scalar has no leading member axis".into(),
            }
        })?;
        if let Self::Stack { axis: 0, inputs } = self {
            return inputs
                .iter()
                .map(|input| {
                    let selected = Self::Stack {
                        axis: 0,
                        inputs: vec![input.clone()],
                    };
                    normalize_bounded_source_ranges(expand_indexed_sources(selected), catalog)
                })
                .collect();
        }
        (0..count)
            .map(|member| {
                self.select_bounded(
                    catalog,
                    TensorSelection::Range {
                        axis: 0,
                        start: member,
                        end: member + 1,
                    },
                )
            })
            .collect()
    }

    /// Selects rows from one matrix while retaining leading singleton axes.
    pub fn select_bounded_matrix_rows<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
        leading_index: usize,
        start: usize,
        end: usize,
    ) -> Result<Self, RecipeError> {
        let metadata = self.infer(catalog)?;
        if metadata.shape.len() < 2 {
            return Err(RecipeError::SelectionPushdownUnsupported {
                operation: "matrix row selection",
                reason: format!("rank {} has no matrix row axis", metadata.shape.len()),
            });
        }
        let row_axis = metadata.shape.len() - 2;
        let leading = usize::try_from(element_count(
            &metadata.shape[..row_axis],
            "leading matrix dimensions",
        )?)
        .map_err(|_| RecipeError::ArithmeticOverflow("leading matrix dimensions"))?;
        if leading_index >= leading {
            return Err(RecipeError::InvalidIndices {
                axis: 0,
                dimension: leading,
            });
        }
        let mut coordinates = vec![0usize; row_axis];
        let mut remainder = leading_index;
        for axis in (0..row_axis).rev() {
            let dimension = metadata.shape[axis];
            coordinates[axis] = remainder % dimension;
            remainder /= dimension;
        }
        let mut selected = self.clone();
        for (axis, coordinate) in coordinates.into_iter().enumerate() {
            selected = selected.select_bounded(
                catalog,
                TensorSelection::Range {
                    axis,
                    start: coordinate,
                    end: coordinate + 1,
                },
            )?;
        }
        selected.select_bounded(
            catalog,
            TensorSelection::Range {
                axis: row_axis,
                start,
                end,
            },
        )
    }

    /// Returns a conservative bound for simultaneously live recipe values.
    pub fn peak_materialization_bytes<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
    ) -> Result<u64, RecipeError> {
        let output_bytes = self.infer(catalog)?.byte_len();
        match self {
            Self::Source { .. } => Ok(output_bytes),
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => {
                let input_bytes = input.infer(catalog)?.byte_len();
                let child_peak = input.peak_materialization_bytes(catalog)?;
                Ok(child_peak.max(input_bytes.checked_add(output_bytes).ok_or(
                    RecipeError::ArithmeticOverflow("unary recipe peak materialization bytes"),
                )?))
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                let mut retained = 0u64;
                let mut peak = 0u64;
                for input in inputs {
                    let child_peak = input.peak_materialization_bytes(catalog)?;
                    peak = peak.max(retained.checked_add(child_peak).ok_or(
                        RecipeError::ArithmeticOverflow("joined recipe child peak bytes"),
                    )?);
                    retained = retained
                        .checked_add(input.infer(catalog)?.byte_len())
                        .ok_or(RecipeError::ArithmeticOverflow(
                            "joined recipe retained input bytes",
                        ))?;
                }
                Ok(peak.max(retained.checked_add(output_bytes).ok_or(
                    RecipeError::ArithmeticOverflow("joined recipe output peak bytes"),
                )?))
            }
        }
    }

    /// Returns every source checkpoint key in deterministic order.
    pub fn source_keys(&self) -> Vec<&str> {
        let mut keys = BTreeSet::new();
        self.collect_source_keys(&mut keys);
        keys.into_iter().collect()
    }

    /// Returns source checkpoint keys in recipe traversal order with repetitions.
    ///
    /// This is the exact source-occurrence contract for consumers whose
    /// execution or identity depends on operand order. Use [`Self::source_keys`]
    /// when only a deterministic unique dependency set is required.
    pub fn source_occurrences(&self) -> Vec<&str> {
        struct Keys<'a>(Vec<&'a str>);
        impl<'a> RecipeSourceVisitorLoan<'a> for Keys<'a> {
            type Error = std::convert::Infallible;
            fn source(&mut self, key: &'a str, _: &'a TensorSelection) -> Result<(), Self::Error> {
                self.0.push(key);
                Ok(())
            }
        }
        let mut keys = Keys(Vec::new());
        match self.visit_source_loans(&mut keys) {
            Ok(()) => keys.0,
            Err(never) => match never {},
        }
    }

    /// Visit actual source declarations without constructing a key/selection Vec.
    /// A failure stops traversal immediately and leaves visitor retirement to its owner.
    pub fn visit_sources<V: RecipeSourceVisitor>(&self, visitor: &mut V) -> Result<(), V::Error> {
        struct Adapter<'v, V>(&'v mut V);
        impl<'a, V: RecipeSourceVisitor> RecipeSourceVisitorLoan<'a> for Adapter<'_, V> {
            type Error = V::Error;
            fn source(
                &mut self,
                key: &'a str,
                selection: &'a TensorSelection,
            ) -> Result<(), Self::Error> {
                self.0.source(key, selection)
            }
            fn enter_join(&mut self) -> Result<(), Self::Error> {
                self.0.enter_join()
            }
            fn leave_join(&mut self) {
                self.0.leave_join()
            }
        }
        self.visit_source_loans(&mut Adapter(visitor))
    }

    fn collect_source_keys<'a>(&'a self, keys: &mut BTreeSet<&'a str>) {
        match self {
            Self::Source { key, .. } => {
                keys.insert(key);
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                for input in inputs {
                    input.collect_source_keys(keys);
                }
            }
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => input.collect_source_keys(keys),
        }
    }

    fn visit_source_loans<'a, V: RecipeSourceVisitorLoan<'a>>(
        &'a self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        match self {
            Self::Source { key, selection } => visitor.source(key, selection),
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                visitor.enter_join()?;
                for input in inputs {
                    input.visit_source_loans(visitor)?;
                }
                visitor.leave_join();
                Ok(())
            }
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => input.visit_source_loans(visitor),
        }
    }

    /// Validates every operation and infers its exact output metadata.
    pub fn infer<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
    ) -> Result<RecipeMetadata, RecipeError> {
        self.with_inferred(catalog, Clone::clone)
    }

    /// Inspects the validated output without cloning a cached successful result.
    /// The callback runs outside the cache mutex. A miss performs the same
    /// inference and retains the same result as [`Self::infer`]; an uncached
    /// catalog owns its temporary metadata until the callback returns. Errors
    /// preserve the ordinary owned error path. This grants no storage custody.
    pub fn with_inferred<C: RecipeCatalog + ?Sized, T>(
        &self,
        catalog: &C,
        inspect: impl FnOnce(&RecipeMetadata) -> T,
    ) -> Result<T, RecipeError> {
        match catalog.recipe_cache() {
            Some(cache) => cache.with_inferred(self, catalog, inspect),
            None => self
                .infer_uncached(catalog)
                .map(|metadata| inspect(&metadata)),
        }
    }

    /// Validates a direct source selection and lends its inferred dimensions.
    /// Prepared catalogs lend their retained metadata. Custom/unprepared sources
    /// use the existing owned metadata API and retain that temporary through the
    /// callback. Selection validation is shared with ordinary recipe inference;
    /// no recipe/key/selection or output shape clone is constructed here.
    /// Unsupported dtype errors retain their ordinary owned diagnostics.
    pub fn with_source_metadata<T>(
        key: &str,
        selection: &TensorSelection,
        source: &dyn CheckpointSource,
        inspect: impl FnOnce(RecipeMetadataView<'_>) -> T,
    ) -> Result<T, RecipeError> {
        if key.trim().is_empty() {
            return Err(RecipeError::EmptySourceKey);
        }
        let owned;
        let metadata = match source.source_metadata_borrowed(key) {
            Ok(metadata) => metadata,
            // Preserve the original source's error semantics and support for
            // custom providers; this fallback is not a closed constructor fit.
            Err(_) => {
                owned = source.source_metadata(key)?;
                &owned
            }
        };
        let shape = SelectedRecipeShape::new(&metadata.logical_shape, selection)?;
        let dtype = RecipeDtype::from(metadata.stored_dtype.clone());
        let byte_len = metadata_byte_len(shape.iter(), &dtype)?;
        Ok(inspect(RecipeMetadataView {
            shape,
            dtype: &dtype,
            byte_len,
        }))
    }

    fn infer_uncached<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
    ) -> Result<RecipeMetadata, RecipeError> {
        match self {
            Self::Source { key, selection } => {
                if key.trim().is_empty() {
                    return Err(RecipeError::EmptySourceKey);
                }
                let metadata = catalog.tensor_metadata(key)?;
                metadata_for(
                    selected_shape(metadata.logical_shape, selection)?,
                    metadata.stored_dtype.into(),
                )
            }
            Self::Select { input, selection } => {
                let metadata = input.infer(catalog)?;
                metadata_for(selected_shape(metadata.shape, selection)?, metadata.dtype)
            }
            Self::Concatenate { axis, inputs } => infer_join(catalog, *axis, inputs, false),
            Self::Stack { axis, inputs } => infer_join(catalog, *axis, inputs, true),
            Self::Reshape { input, shape } => {
                let metadata = input.infer(catalog)?;
                validate_reshape(&metadata.shape, shape)?;
                metadata_for(shape.clone(), metadata.dtype)
            }
            Self::Transpose { input, axes } => {
                let metadata = input.infer(catalog)?;
                validate_permutation(axes, metadata.shape.len())?;
                metadata_for(
                    axes.iter().map(|axis| metadata.shape[*axis]).collect(),
                    metadata.dtype,
                )
            }
            Self::Cast { input, dtype } => metadata_for(input.infer(catalog)?.shape, dtype.clone()),
            Self::View {
                input,
                dtype,
                shape,
            } => {
                let input = input.infer(catalog)?;
                let output = metadata_for(shape.clone(), dtype.clone())?;
                if input.byte_len != output.byte_len {
                    return Err(RecipeError::ByteCountMismatch {
                        input: input.byte_len,
                        output: output.byte_len,
                    });
                }
                Ok(output)
            }
            Self::NegLog { input } | Self::SubtractOne { input } => input.infer(catalog),
        }
    }
}

impl<'a> SelectedRecipeShape<'a> {
    fn new(shape: &'a [usize], selection: &'a TensorSelection) -> Result<Self, RecipeError> {
        let mut output = Self {
            dimensions: shape,
            replacement: None,
        };
        match selection {
            TensorSelection::Full => {}
            TensorSelection::Range { axis, start, end } => {
                let rank = shape.len();
                let dimension = shape
                    .get(*axis)
                    .ok_or(RecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
                if start >= end || *end > *dimension {
                    return Err(RecipeError::InvalidRange {
                        axis: *axis,
                        start: *start,
                        end: *end,
                        dimension: *dimension,
                    });
                }
                output.replacement = Some((*axis, end - start));
            }
            TensorSelection::Indices { axis, indices } => {
                let rank = shape.len();
                let dimension = shape
                    .get(*axis)
                    .ok_or(RecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
                if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                    return Err(RecipeError::InvalidIndices {
                        axis: *axis,
                        dimension: *dimension,
                    });
                }
                output.replacement = Some((*axis, indices.len()));
            }
            TensorSelection::Contiguous {
                offset_elements,
                shape: selected,
            } => {
                if selected.is_empty() || selected.contains(&0) {
                    return Err(RecipeError::InvalidContiguousSelection);
                }
                let full = element_count(shape, "contiguous source")?;
                let count = element_count(selected, "contiguous selection")?;
                let end = u64::try_from(*offset_elements)
                    .map_err(|_| RecipeError::ArithmeticOverflow("contiguous offset"))?
                    .checked_add(count)
                    .ok_or(RecipeError::ArithmeticOverflow("contiguous end"))?;
                if end > full {
                    return Err(RecipeError::InvalidContiguousSelection);
                }
                output.dimensions = selected;
            }
        }
        Ok(output)
    }

    fn iter(self) -> impl ExactSizeIterator<Item = usize> + Clone + 'a {
        self.dimensions
            .iter()
            .copied()
            .enumerate()
            .map(move |(axis, value)| {
                self.replacement
                    .filter(|(changed, _)| *changed == axis)
                    .map_or(value, |(_, dimension)| dimension)
            })
    }
}

fn selected_shape(
    mut shape: Vec<usize>,
    selection: &TensorSelection,
) -> Result<Vec<usize>, RecipeError> {
    let selected = SelectedRecipeShape::new(&shape, selection)?;
    let replacement = selected.replacement;
    if let TensorSelection::Contiguous { shape: output, .. } = selection {
        return Ok(output.clone());
    }
    if let Some((axis, dimension)) = replacement {
        shape[axis] = dimension;
    }
    Ok(shape)
}

fn expand_indexed_sources(recipe: DerivedWeightRecipe) -> DerivedWeightRecipe {
    match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: TensorSelection::Indices { axis, indices },
        } => {
            let mut runs = Vec::<(usize, usize)>::new();
            for index in indices {
                if let Some((_, end)) = runs.last_mut() {
                    if *end == index {
                        *end += 1;
                        continue;
                    }
                }
                runs.push((index, index + 1));
            }
            let mut inputs = runs
                .into_iter()
                .map(|(start, end)| {
                    DerivedWeightRecipe::source(
                        key.clone(),
                        TensorSelection::Range { axis, start, end },
                    )
                })
                .collect::<Vec<_>>();
            if inputs.len() == 1 {
                inputs.pop().unwrap()
            } else {
                DerivedWeightRecipe::Concatenate { axis, inputs }
            }
        }
        DerivedWeightRecipe::Source { .. } => recipe,
        DerivedWeightRecipe::Select { input, selection } => DerivedWeightRecipe::Select {
            input: Box::new(expand_indexed_sources(*input)),
            selection,
        },
        DerivedWeightRecipe::Concatenate { axis, inputs } => DerivedWeightRecipe::Concatenate {
            axis,
            inputs: inputs.into_iter().map(expand_indexed_sources).collect(),
        },
        DerivedWeightRecipe::Stack { axis, inputs } => DerivedWeightRecipe::Stack {
            axis,
            inputs: inputs.into_iter().map(expand_indexed_sources).collect(),
        },
        DerivedWeightRecipe::Reshape { input, shape } => DerivedWeightRecipe::Reshape {
            input: Box::new(expand_indexed_sources(*input)),
            shape,
        },
        DerivedWeightRecipe::Transpose { input, axes } => DerivedWeightRecipe::Transpose {
            input: Box::new(expand_indexed_sources(*input)),
            axes,
        },
        DerivedWeightRecipe::Cast { input, dtype } => DerivedWeightRecipe::Cast {
            input: Box::new(expand_indexed_sources(*input)),
            dtype,
        },
        DerivedWeightRecipe::View {
            input,
            dtype,
            shape,
        } => DerivedWeightRecipe::View {
            input: Box::new(expand_indexed_sources(*input)),
            dtype,
            shape,
        },
        DerivedWeightRecipe::NegLog { input } => DerivedWeightRecipe::NegLog {
            input: Box::new(expand_indexed_sources(*input)),
        },
        DerivedWeightRecipe::SubtractOne { input } => DerivedWeightRecipe::SubtractOne {
            input: Box::new(expand_indexed_sources(*input)),
        },
    }
}

fn normalize_bounded_source_ranges<C: RecipeCatalog + ?Sized>(
    recipe: DerivedWeightRecipe,
    store: &C,
) -> Result<DerivedWeightRecipe, RecipeError> {
    Ok(match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: TensorSelection::Range { axis, start, end },
        } if axis > 0 => {
            let shape = store.tensor_metadata(&key)?.logical_shape;
            if shape[..axis].iter().product::<usize>() == 1 {
                let trailing = shape[axis + 1..]
                    .iter()
                    .try_fold(1usize, |count, dimension| {
                        count
                            .checked_mul(*dimension)
                            .ok_or(RecipeError::ArithmeticOverflow(
                                "bounded source range trailing span",
                            ))
                    })?;
                let offset_elements =
                    start
                        .checked_mul(trailing)
                        .ok_or(RecipeError::ArithmeticOverflow(
                            "bounded source range offset",
                        ))?;
                let mut selected_shape = shape;
                selected_shape[axis] = end - start;
                DerivedWeightRecipe::source(
                    key,
                    TensorSelection::Contiguous {
                        offset_elements,
                        shape: selected_shape,
                    },
                )
            } else {
                DerivedWeightRecipe::source(key, TensorSelection::Range { axis, start, end })
            }
        }
        DerivedWeightRecipe::Source { .. } => recipe,
        DerivedWeightRecipe::Select { input, selection } => DerivedWeightRecipe::Select {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            selection,
        },
        DerivedWeightRecipe::Concatenate { axis, inputs } => DerivedWeightRecipe::Concatenate {
            axis,
            inputs: inputs
                .into_iter()
                .map(|input| normalize_bounded_source_ranges(input, store))
                .collect::<Result<Vec<_>, _>>()?,
        },
        DerivedWeightRecipe::Stack { axis, inputs } => DerivedWeightRecipe::Stack {
            axis,
            inputs: inputs
                .into_iter()
                .map(|input| normalize_bounded_source_ranges(input, store))
                .collect::<Result<Vec<_>, _>>()?,
        },
        DerivedWeightRecipe::Reshape { input, shape } => DerivedWeightRecipe::Reshape {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            shape,
        },
        DerivedWeightRecipe::Transpose { input, axes } => DerivedWeightRecipe::Transpose {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            axes,
        },
        DerivedWeightRecipe::Cast { input, dtype } => DerivedWeightRecipe::Cast {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            dtype,
        },
        DerivedWeightRecipe::View {
            input,
            dtype,
            shape,
        } => DerivedWeightRecipe::View {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            dtype,
            shape,
        },
        DerivedWeightRecipe::NegLog { input } => DerivedWeightRecipe::NegLog {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
        },
        DerivedWeightRecipe::SubtractOne { input } => DerivedWeightRecipe::SubtractOne {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
        },
    })
}

fn push_selection<C: RecipeCatalog + ?Sized>(
    recipe: &DerivedWeightRecipe,
    store: &C,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, RecipeError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok(recipe.clone());
    }
    match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: source_selection,
        } => {
            let source_shape = store.tensor_metadata(key)?.logical_shape;
            let source_selection = normalize_selection(source_selection.clone(), &source_shape)?;
            let selected_source_shape = selected_shape(source_shape.clone(), &source_selection)?;
            let selection = normalize_selection(selection, &selected_source_shape)?;
            if matches!(selection, TensorSelection::Full) {
                return Ok(DerivedWeightRecipe::source(key.clone(), source_selection));
            }
            if matches!(source_selection, TensorSelection::Full) {
                return Ok(DerivedWeightRecipe::source(key.clone(), selection));
            }
            if let Some(selection) = select_from_contiguous_span(&source_selection, &selection)? {
                return Ok(DerivedWeightRecipe::source(key.clone(), selection));
            }
            if selection_axis(&source_selection) == selection_axis(&selection) {
                return Ok(DerivedWeightRecipe::source(
                    key.clone(),
                    compose_same_axis_selection(&source_selection, &selection)?,
                ));
            }
            if let Some(contiguous) =
                combine_independent_ranges(&source_shape, &source_selection, &selection)?
            {
                return Ok(DerivedWeightRecipe::source(key.clone(), contiguous));
            }
            // Keep the leading-axis restriction closest to the physical source.
            // Later row tiles can then combine with an expert/member restriction
            // before any independent column selection is applied.
            let (source_selection, selection) = match (
                selection_axis(&source_selection),
                selection_axis(&selection),
            ) {
                (Some(existing), Some(requested)) if requested < existing => {
                    (selection, source_selection)
                }
                _ => (source_selection, selection),
            };
            Ok(DerivedWeightRecipe::Select {
                input: Box::new(DerivedWeightRecipe::source(key.clone(), source_selection)),
                selection,
            })
        }
        DerivedWeightRecipe::Select {
            input,
            selection: existing,
        } => {
            if matches!(existing, TensorSelection::Full) {
                return push_selection(input, store, selection);
            }
            if selection_axis(existing) == selection_axis(&selection) {
                return push_selection(
                    input,
                    store,
                    compose_same_axis_selection(existing, &selection)?,
                );
            }
            // Independent axes commute. Push the lower axis first so member and
            // row restrictions can form a contiguous span before selecting columns.
            let (inner, outer) = match (selection_axis(existing), selection_axis(&selection)) {
                (Some(existing_axis), Some(requested_axis)) if existing_axis < requested_axis => {
                    (existing.clone(), selection)
                }
                _ => (selection, existing.clone()),
            };
            let selected_input = push_selection(input, store, inner)?;
            if matches!(selected_input, DerivedWeightRecipe::Source { .. }) {
                // Source composition cannot recurse into another Select. Collapse
                // the outer range too when the inner restriction made it contiguous.
                push_selection(&selected_input, store, outer)
            } else {
                // Do not recurse into the newly produced tree: noncontiguous axes
                // could otherwise swap indefinitely. Recursive work above always
                // descends into the original, strictly smaller input recipe.
                Ok(DerivedWeightRecipe::Select {
                    input: Box::new(selected_input),
                    selection: outer,
                })
            }
        }
        DerivedWeightRecipe::Concatenate { axis, inputs } => {
            push_concatenate_selection(*axis, inputs, store, selection)
        }
        DerivedWeightRecipe::Stack { axis, inputs } => {
            push_stack_selection(*axis, inputs, store, selection)
        }
        DerivedWeightRecipe::Reshape { input, shape } => {
            let input_metadata = input.infer(store)?;
            let output_metadata = recipe.infer(store)?;
            let input_selection = map_reinterpret_selection(
                &input_metadata,
                &output_metadata,
                &selection,
                "reshape",
            )?;
            Ok(DerivedWeightRecipe::Reshape {
                input: Box::new(push_selection(input, store, input_selection)?),
                shape: selected_shape(shape.clone(), &selection)?,
            })
        }
        DerivedWeightRecipe::Transpose { input, axes } => {
            let output_axis = selection_axis(&selection).expect("non-full selection");
            let input_axis = *axes
                .get(output_axis)
                .ok_or(RecipeError::InvalidSelectionAxis {
                    axis: output_axis,
                    rank: axes.len(),
                })?;
            Ok(DerivedWeightRecipe::Transpose {
                input: Box::new(push_selection(
                    input,
                    store,
                    selection_with_axis(selection, input_axis),
                )?),
                axes: axes.clone(),
            })
        }
        DerivedWeightRecipe::Cast { input, dtype } => Ok(DerivedWeightRecipe::Cast {
            input: Box::new(push_selection(input, store, selection)?),
            dtype: dtype.clone(),
        }),
        DerivedWeightRecipe::View {
            input,
            dtype,
            shape,
        } => {
            let input_metadata = input.infer(store)?;
            let output_metadata = recipe.infer(store)?;
            let input_selection =
                map_reinterpret_selection(&input_metadata, &output_metadata, &selection, "view")?;
            Ok(DerivedWeightRecipe::View {
                input: Box::new(push_selection(input, store, input_selection)?),
                dtype: dtype.clone(),
                shape: selected_shape(shape.clone(), &selection)?,
            })
        }
        DerivedWeightRecipe::NegLog { input } => Ok(DerivedWeightRecipe::NegLog {
            input: Box::new(push_selection(input, store, selection)?),
        }),
        DerivedWeightRecipe::SubtractOne { input } => Ok(DerivedWeightRecipe::SubtractOne {
            input: Box::new(push_selection(input, store, selection)?),
        }),
    }
}

fn push_concatenate_selection<C: RecipeCatalog + ?Sized>(
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    store: &C,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, RecipeError> {
    let selected_axis = selection_axis(&selection).expect("non-full selection");
    if selected_axis != axis {
        let inputs = inputs
            .iter()
            .map(|input| push_selection(input, store, selection.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(DerivedWeightRecipe::Concatenate { axis, inputs });
    }
    let metadata = inputs
        .iter()
        .map(|input| input.infer(store))
        .collect::<Result<Vec<_>, _>>()?;
    let dimensions = metadata
        .iter()
        .map(|item| item.shape[axis])
        .collect::<Vec<_>>();
    let mut rewritten = Vec::new();
    match selection {
        TensorSelection::Range { start, end, .. } => {
            let mut offset = 0usize;
            for (input, dimension) in inputs.iter().zip(dimensions) {
                let child_end =
                    offset
                        .checked_add(dimension)
                        .ok_or(RecipeError::ArithmeticOverflow(
                            "concatenate selection offset",
                        ))?;
                let overlap_start = start.max(offset);
                let overlap_end = end.min(child_end);
                if overlap_start < overlap_end {
                    let child_selection = normalize_selection(
                        TensorSelection::Range {
                            axis,
                            start: overlap_start - offset,
                            end: overlap_end - offset,
                        },
                        &input.infer(store)?.shape,
                    )?;
                    rewritten.push(push_selection(input, store, child_selection)?);
                }
                offset = child_end;
            }
        }
        TensorSelection::Indices { indices, .. } => {
            let mut offsets = Vec::with_capacity(dimensions.len() + 1);
            offsets.push(0usize);
            for dimension in dimensions {
                let next = offsets
                    .last()
                    .copied()
                    .unwrap()
                    .checked_add(dimension)
                    .ok_or(RecipeError::ArithmeticOverflow(
                        "concatenate selection offset",
                    ))?;
                offsets.push(next);
            }
            let mut runs = Vec::<(usize, Vec<usize>)>::new();
            for index in indices {
                let child = offsets
                    .windows(2)
                    .position(|bounds| index >= bounds[0] && index < bounds[1])
                    .ok_or(RecipeError::InvalidIndices {
                        axis,
                        dimension: *offsets.last().unwrap(),
                    })?;
                let local = index - offsets[child];
                if let Some((last_child, local_indices)) = runs.last_mut() {
                    if *last_child == child {
                        local_indices.push(local);
                        continue;
                    }
                }
                runs.push((child, vec![local]));
            }
            for (child, indices) in runs {
                rewritten.push(push_selection(
                    &inputs[child],
                    store,
                    TensorSelection::Indices { axis, indices },
                )?);
            }
        }
        TensorSelection::Full => unreachable!(),
        TensorSelection::Contiguous { .. } => {
            return Err(RecipeError::SelectionPushdownUnsupported {
                operation: "concatenate",
                reason: "a storage-contiguous span has no concatenate-axis semantics".into(),
            });
        }
    }
    match rewritten.len() {
        0 => Err(RecipeError::SelectionPushdownUnsupported {
            operation: "concatenate",
            reason: "selection did not intersect any child".into(),
        }),
        1 => Ok(rewritten.pop().unwrap()),
        _ => Ok(DerivedWeightRecipe::Concatenate {
            axis,
            inputs: rewritten,
        }),
    }
}

fn push_stack_selection<C: RecipeCatalog + ?Sized>(
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    store: &C,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, RecipeError> {
    let selected_axis = selection_axis(&selection).expect("non-full selection");
    if selected_axis == axis {
        let selected = match selection {
            TensorSelection::Range { start, end, .. } => inputs[start..end].to_vec(),
            TensorSelection::Indices { indices, .. } => indices
                .into_iter()
                .map(|index| inputs[index].clone())
                .collect(),
            TensorSelection::Full => unreachable!(),
            TensorSelection::Contiguous { .. } => {
                return Err(RecipeError::SelectionPushdownUnsupported {
                    operation: "stack",
                    reason: "a storage-contiguous span has no stack-axis semantics".into(),
                });
            }
        };
        return Ok(DerivedWeightRecipe::Stack {
            axis,
            inputs: selected,
        });
    }
    let input_axis = if selected_axis < axis {
        selected_axis
    } else {
        selected_axis - 1
    };
    let input_selection = selection_with_axis(selection, input_axis);
    let inputs = inputs
        .iter()
        .map(|input| push_selection(input, store, input_selection.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DerivedWeightRecipe::Stack { axis, inputs })
}

fn map_reinterpret_selection(
    input: &RecipeMetadata,
    output: &RecipeMetadata,
    selection: &TensorSelection,
    operation: &'static str,
) -> Result<TensorSelection, RecipeError> {
    let output_axis = selection_axis(selection).expect("non-full selection");
    let output_unit = axis_unit_bytes(&output.shape, output.dtype.bit_width()?, output_axis)?;
    let output_cycle = output_unit
        .checked_mul(output.shape[output_axis] as u64)
        .ok_or(RecipeError::ArithmeticOverflow("selection output cycle"))?;
    for input_axis in 0..input.shape.len() {
        let input_unit = axis_unit_bytes(&input.shape, input.dtype.bit_width()?, input_axis)?;
        let input_cycle = input_unit
            .checked_mul(input.shape[input_axis] as u64)
            .ok_or(RecipeError::ArithmeticOverflow("selection input cycle"))?;
        if input_cycle != output_cycle {
            continue;
        }
        if let Some(mapped) = map_selection_units(
            selection,
            input_axis,
            input.shape[input_axis],
            output_unit,
            input_unit,
        )? {
            return normalize_selection(mapped, &input.shape);
        }
    }
    Err(RecipeError::SelectionPushdownUnsupported {
        operation,
        reason: format!(
            "axis {output_axis} selection cannot be expressed as a single-axis bounded selection from shape {:?} to {:?}",
            input.shape, output.shape
        ),
    })
}

fn map_selection_units(
    selection: &TensorSelection,
    input_axis: usize,
    input_dimension: usize,
    output_unit: u64,
    input_unit: u64,
) -> Result<Option<TensorSelection>, RecipeError> {
    let map_interval = |start: usize, end: usize| -> Result<Option<(usize, usize)>, RecipeError> {
        let start_bytes = (start as u64)
            .checked_mul(output_unit)
            .ok_or(RecipeError::ArithmeticOverflow("selection interval start"))?;
        let end_bytes = (end as u64)
            .checked_mul(output_unit)
            .ok_or(RecipeError::ArithmeticOverflow("selection interval end"))?;
        if start_bytes % input_unit != 0 || end_bytes % input_unit != 0 {
            return Ok(None);
        }
        let start = usize::try_from(start_bytes / input_unit)
            .map_err(|_| RecipeError::ArithmeticOverflow("mapped selection start"))?;
        let end = usize::try_from(end_bytes / input_unit)
            .map_err(|_| RecipeError::ArithmeticOverflow("mapped selection end"))?;
        Ok((end <= input_dimension).then_some((start, end)))
    };
    match selection {
        TensorSelection::Range { start, end, .. } => Ok(map_interval(*start, *end)?.map(
            |(start, end)| TensorSelection::Range {
                axis: input_axis,
                start,
                end,
            },
        )),
        TensorSelection::Indices { indices, .. } => {
            let mut mapped = Vec::new();
            let mut run_start = indices[0];
            let mut run_end = run_start + 1;
            for index in indices.iter().copied().skip(1) {
                if index == run_end {
                    run_end += 1;
                    continue;
                }
                let Some((start, end)) = map_interval(run_start, run_end)? else {
                    return Ok(None);
                };
                mapped.extend(start..end);
                run_start = index;
                run_end = index + 1;
            }
            let Some((start, end)) = map_interval(run_start, run_end)? else {
                return Ok(None);
            };
            mapped.extend(start..end);
            Ok(Some(TensorSelection::Indices {
                axis: input_axis,
                indices: mapped,
            }))
        }
        TensorSelection::Full => Ok(Some(TensorSelection::Full)),
        TensorSelection::Contiguous { .. } => Ok(None),
    }
}

fn axis_unit_bytes(shape: &[usize], dtype_width: u64, axis: usize) -> Result<u64, RecipeError> {
    shape[axis + 1..]
        .iter()
        .try_fold(dtype_width, |bytes, dimension| {
            bytes
                .checked_mul(*dimension as u64)
                .ok_or(RecipeError::ArithmeticOverflow("selection axis unit"))
        })
}

fn selection_axis(selection: &TensorSelection) -> Option<usize> {
    match selection {
        TensorSelection::Full => None,
        TensorSelection::Range { axis, .. } | TensorSelection::Indices { axis, .. } => Some(*axis),
        TensorSelection::Contiguous { .. } => None,
    }
}

fn selection_with_axis(selection: TensorSelection, axis: usize) -> TensorSelection {
    match selection {
        TensorSelection::Full => TensorSelection::Full,
        TensorSelection::Range { start, end, .. } => TensorSelection::Range { axis, start, end },
        TensorSelection::Indices { indices, .. } => TensorSelection::Indices { axis, indices },
        selection @ TensorSelection::Contiguous { .. } => selection,
    }
}

fn normalize_selection(
    selection: TensorSelection,
    shape: &[usize],
) -> Result<TensorSelection, RecipeError> {
    selected_shape(shape.to_vec(), &selection)?;
    match selection {
        TensorSelection::Range {
            axis,
            start: 0,
            end,
        } if end == shape[axis] => Ok(TensorSelection::Full),
        TensorSelection::Indices { axis, indices }
            if indices.windows(2).all(|pair| pair[1] == pair[0] + 1) =>
        {
            let start = indices[0];
            let end = indices[indices.len() - 1] + 1;
            if start == 0 && end == shape[axis] {
                Ok(TensorSelection::Full)
            } else {
                Ok(TensorSelection::Range { axis, start, end })
            }
        }
        selection => Ok(selection),
    }
}

fn compose_same_axis_selection(
    existing: &TensorSelection,
    requested: &TensorSelection,
) -> Result<TensorSelection, RecipeError> {
    debug_assert_eq!(selection_axis(existing), selection_axis(requested));
    let axis = selection_axis(existing).expect("non-full selections");
    match (existing, requested) {
        (
            TensorSelection::Range { start, .. },
            TensorSelection::Range {
                start: requested_start,
                end: requested_end,
                ..
            },
        ) => Ok(TensorSelection::Range {
            axis,
            start: start + requested_start,
            end: start + requested_end,
        }),
        (TensorSelection::Range { start, .. }, TensorSelection::Indices { indices, .. }) => {
            Ok(TensorSelection::Indices {
                axis,
                indices: indices.iter().map(|index| start + index).collect(),
            })
        }
        (TensorSelection::Indices { indices, .. }, TensorSelection::Range { start, end, .. }) => {
            Ok(TensorSelection::Indices {
                axis,
                indices: indices[*start..*end].to_vec(),
            })
        }
        (
            TensorSelection::Indices {
                indices: existing, ..
            },
            TensorSelection::Indices { indices, .. },
        ) => Ok(TensorSelection::Indices {
            axis,
            indices: indices.iter().map(|index| existing[*index]).collect(),
        }),
        _ => Err(RecipeError::SelectionPushdownUnsupported {
            operation: "selection composition",
            reason: "full selections must be normalized before composition".into(),
        }),
    }
}

fn combine_independent_ranges(
    source_shape: &[usize],
    existing: &TensorSelection,
    requested: &TensorSelection,
) -> Result<Option<TensorSelection>, RecipeError> {
    let (
        TensorSelection::Range {
            axis: existing_axis,
            start: existing_start,
            end: existing_end,
        },
        TensorSelection::Range {
            axis: requested_axis,
            start: requested_start,
            end: requested_end,
        },
    ) = (existing, requested)
    else {
        return Ok(None);
    };
    if existing_axis == requested_axis {
        return Ok(None);
    }
    let mut starts = vec![0usize; source_shape.len()];
    let mut ends = source_shape.to_vec();
    starts[*existing_axis] = *existing_start;
    ends[*existing_axis] = *existing_end;
    starts[*requested_axis] = *requested_start;
    ends[*requested_axis] = *requested_end;
    let selected_shape = starts
        .iter()
        .zip(&ends)
        .map(|(start, end)| end - start)
        .collect::<Vec<_>>();
    let Some(last_partial) = (0..source_shape.len())
        .rev()
        .find(|axis| starts[*axis] != 0 || ends[*axis] != source_shape[*axis])
    else {
        return Ok(Some(TensorSelection::Full));
    };
    if selected_shape[..last_partial]
        .iter()
        .any(|dimension| *dimension != 1)
        || (last_partial + 1..source_shape.len())
            .any(|axis| starts[axis] != 0 || ends[axis] != source_shape[axis])
    {
        return Ok(None);
    }
    let mut offset_elements = 0usize;
    let mut stride = 1usize;
    for axis in (0..source_shape.len()).rev() {
        offset_elements = offset_elements
            .checked_add(starts[axis].checked_mul(stride).ok_or(
                RecipeError::ArithmeticOverflow("contiguous selection offset"),
            )?)
            .ok_or(RecipeError::ArithmeticOverflow(
                "contiguous selection offset",
            ))?;
        stride = stride
            .checked_mul(source_shape[axis])
            .ok_or(RecipeError::ArithmeticOverflow(
                "contiguous selection stride",
            ))?;
    }
    Ok(Some(TensorSelection::Contiguous {
        offset_elements,
        shape: selected_shape,
    }))
}

fn select_from_contiguous_span(
    existing: &TensorSelection,
    requested: &TensorSelection,
) -> Result<Option<TensorSelection>, RecipeError> {
    let TensorSelection::Contiguous {
        offset_elements,
        shape,
    } = existing
    else {
        return Ok(None);
    };
    let (axis, start, end) = match requested {
        TensorSelection::Range { axis, start, end } => (*axis, *start, *end),
        TensorSelection::Indices { axis, indices }
            if indices.windows(2).all(|pair| pair[1] == pair[0] + 1) =>
        {
            (*axis, indices[0], indices[indices.len() - 1] + 1)
        }
        _ => return Ok(None),
    };
    if shape[..axis].iter().product::<usize>() != 1 {
        return Ok(None);
    }
    let trailing = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(RecipeError::ArithmeticOverflow(
                    "contiguous selection trailing span",
                ))
        })?;
    let offset_elements = offset_elements
        .checked_add(
            start
                .checked_mul(trailing)
                .ok_or(RecipeError::ArithmeticOverflow(
                    "contiguous selection offset",
                ))?,
        )
        .ok_or(RecipeError::ArithmeticOverflow(
            "contiguous selection offset",
        ))?;
    let mut selected_shape = shape.clone();
    selected_shape[axis] = end - start;
    Ok(Some(TensorSelection::Contiguous {
        offset_elements,
        shape: selected_shape,
    }))
}

fn infer_join<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    stack: bool,
) -> Result<RecipeMetadata, RecipeError> {
    if inputs.is_empty() {
        return Err(RecipeError::EmptyInputs);
    }
    let metadata = inputs
        .iter()
        .map(|input| input.infer(catalog))
        .collect::<Result<Vec<_>, _>>()?;
    let mut shape = Vec::with_capacity(metadata[0].shape.len() + usize::from(stack));
    fill_join_shape(axis, metadata.iter(), stack, &mut shape)?;
    metadata_for(shape, metadata[0].dtype.clone())
}

fn validate_reshape(input: &[usize], output: &[usize]) -> Result<(), RecipeError> {
    let old_count = element_count(input, "reshape input")?;
    let new_count = element_count(output, "reshape output")?;
    if old_count != new_count {
        return Err(RecipeError::ElementCountMismatch {
            input: old_count,
            output: new_count,
        });
    }
    Ok(())
}
fn validate_permutation(axes: &[usize], rank: usize) -> Result<(), RecipeError> {
    if axes.len() != rank
        || axes
            .iter()
            .enumerate()
            .any(|(index, axis)| *axis >= rank || axes[..index].contains(axis))
    {
        return Err(RecipeError::InvalidPermutation {
            axes: axes.to_vec(),
            rank,
        });
    }
    Ok(())
}
fn fill_join_shape<'a>(
    axis: usize,
    metadata: impl Iterator<Item = &'a RecipeMetadata> + Clone,
    stack: bool,
    shape: &mut Vec<usize>,
) -> Result<(), RecipeError> {
    let first = metadata.clone().next().ok_or(RecipeError::EmptyInputs)?;
    if metadata.clone().any(|item| item.dtype != first.dtype) {
        return Err(RecipeError::DtypeMismatch);
    }
    let rank = first.shape.len();
    if axis > rank || (!stack && axis == rank) {
        return Err(RecipeError::InvalidJoinAxis { axis, rank, stack });
    }
    let output_rank = rank
        .checked_add(usize::from(stack))
        .ok_or(RecipeError::ArithmeticOverflow("join output rank"))?;
    if output_rank > shape.capacity() {
        return Err(RecipeError::ArithmeticOverflow("join shape destination"));
    }
    if stack {
        if metadata.clone().any(|item| item.shape != first.shape) {
            return Err(RecipeError::ShapeMismatch);
        }
        shape.extend_from_slice(&first.shape);
        shape.insert(axis, metadata.count());
    } else {
        shape.extend_from_slice(&first.shape);
        shape[axis] = 0;
        for item in metadata {
            if item.shape.len() != rank
                || item
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(index, dimension)| index != axis && *dimension != first.shape[index])
            {
                return Err(RecipeError::ShapeMismatch);
            }
            shape[axis] = shape[axis]
                .checked_add(item.shape[axis])
                .ok_or(RecipeError::ArithmeticOverflow("concatenate dimension"))?;
        }
    }
    Ok(())
}

fn metadata_for(shape: Vec<usize>, dtype: RecipeDtype) -> Result<RecipeMetadata, RecipeError> {
    let byte_len = metadata_byte_len(shape.iter().copied(), &dtype)?;
    Ok(RecipeMetadata {
        shape,
        dtype,
        byte_len,
    })
}

fn metadata_byte_len(
    mut shape: impl Iterator<Item = usize>,
    dtype: &RecipeDtype,
) -> Result<u64, RecipeError> {
    let count = shape.try_fold(1u64, |count, dimension| {
        count
            .checked_mul(
                u64::try_from(dimension)
                    .map_err(|_| RecipeError::ArithmeticOverflow("recipe output"))?,
            )
            .ok_or(RecipeError::ArithmeticOverflow("recipe output"))
    })?;
    let bits = count
        .checked_mul(dtype.bit_width()?)
        .ok_or(RecipeError::ArithmeticOverflow("recipe output bits"))?;
    let byte_len = bits
        .checked_add(7)
        .ok_or(RecipeError::ArithmeticOverflow("recipe output bytes"))?
        / 8;
    if byte_len == 0 {
        return Err(RecipeError::ZeroSizedOutput);
    }
    Ok(byte_len)
}

fn element_count(shape: &[usize], context: &'static str) -> Result<u64, RecipeError> {
    shape.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(
                u64::try_from(*dimension).map_err(|_| RecipeError::ArithmeticOverflow(context))?,
            )
            .ok_or(RecipeError::ArithmeticOverflow(context))
    })
}

/// Structured neutral recipe validation failures.
#[derive(Debug, Clone, thiserror::Error)]
#[allow(missing_docs)]
pub enum RecipeError {
    #[error(transparent)]
    EncodedSelection(#[from] crate::store::SafetensorsReadError<'static>),
    #[error("encoded source occurrence {0} disagrees with its catalog entry")]
    InconsistentReadSource(usize),
    #[error("encoded recipe projection reserve failed")]
    ProjectionReserve(#[source] std::collections::TryReserveError),
    #[error("finite encoded-read inference is unavailable")]
    InferenceUnavailable,
    #[error("finite encoded-read inference reserve failed")]
    InferenceReserve(#[source] std::collections::TryReserveError),
    #[error("encoded read catalog: {0}")]
    ReadCatalog(#[from] ReadBatchCatalogBuildError<()>),
    #[error("derived-weight source key must not be empty")]
    EmptySourceKey,
    #[error("derived-weight output name must not be empty")]
    EmptyOutputName,
    #[error("derived-weight recipe family requires at least one output")]
    EmptyOutputs,
    #[error("derived-weight output {target:?} is declared more than once")]
    DuplicateOutput { target: String },
    #[error("derived-weight alias name must not be empty")]
    EmptyAliasName,
    #[error("derived-weight alias {alias:?} is declared more than once")]
    DuplicateAlias { alias: String },
    #[error("derived-weight alias {alias:?} collides with a canonical output")]
    AliasOutputCollision { alias: String },
    #[error("derived-weight alias {alias:?} has unknown destination {destination:?}")]
    InvalidAliasDestination { alias: String, destination: String },
    #[error("derived-weight alias cycle contains {alias:?}")]
    AliasCycle { alias: String },
    #[error("matrix-family affine biases require a scale companion")]
    MatrixBiasWithoutScales,
    #[error("matrix-family weight must have rank at least two, got {shape:?}")]
    InvalidMatrixFamilyWeight { shape: Vec<usize> },
    #[error(
        "matrix-family {member} geometry {companion:?} is incompatible with weight {weight:?}"
    )]
    MatrixCompanionGeometry {
        member: &'static str,
        weight: Vec<usize>,
        companion: Vec<usize>,
    },
    #[error("matrix-family scale geometry {scales:?} differs from affine biases {biases:?}")]
    MatrixScaleBiasGeometry {
        scales: Vec<usize>,
        biases: Vec<usize>,
    },
    #[error("matrix-family leading selection must use axis 0, got axis {axis}")]
    MatrixFamilySelectionAxis { axis: usize },
    #[error("matrix-family leading selection does not accept a scalar contiguous span")]
    MatrixFamilyContiguousSelection,
    #[error("fused source {tensor:?} requires positive, named output segments")]
    InvalidFusedSplit { tensor: String },
    #[error(
        "fused source {tensor:?} axis {axis} has dimension {dimension}, but output widths sum to {outputs}"
    )]
    FusedSplitWidthMismatch {
        tensor: String,
        axis: usize,
        dimension: usize,
        outputs: usize,
    },
    #[error("selection axis {axis} is outside rank {rank}")]
    InvalidSelectionAxis { axis: usize, rank: usize },
    #[error("range {start}..{end} is invalid for axis {axis} dimension {dimension}")]
    InvalidRange {
        axis: usize,
        start: usize,
        end: usize,
        dimension: usize,
    },
    #[error("ordered indices are empty or outside axis {axis} dimension {dimension}")]
    InvalidIndices { axis: usize, dimension: usize },
    #[error("contiguous selection is empty or outside its source tensor")]
    InvalidContiguousSelection,
    #[error("concatenate and stack recipes require at least one input")]
    EmptyInputs,
    #[error("derived-weight inputs have different dtypes")]
    DtypeMismatch,
    #[error("derived-weight inputs have incompatible shapes")]
    ShapeMismatch,
    #[error("axis {axis} is invalid for rank {rank} (stack={stack})")]
    InvalidJoinAxis {
        axis: usize,
        rank: usize,
        stack: bool,
    },
    #[error("reshape changes element count from {input} to {output}")]
    ElementCountMismatch { input: u64, output: u64 },
    #[error("bitwise view changes byte count from {input} to {output}")]
    ByteCountMismatch { input: u64, output: u64 },
    #[error("axes {axes:?} are not a permutation of rank {rank}")]
    InvalidPermutation { axes: Vec<usize>, rank: usize },
    #[error("derived-weight output must contain at least one byte")]
    ZeroSizedOutput,
    #[error("derived-weight dtype {dtype} is unsupported")]
    UnsupportedDtype { dtype: String },
    #[error("derived-weight arithmetic overflow: {0}")]
    ArithmeticOverflow(&'static str),
    #[error("cannot push selection through {operation}: {reason}")]
    SelectionPushdownUnsupported {
        operation: &'static str,
        reason: String,
    },
    #[error(transparent)]
    Store(#[from] StoreError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{EncodedTensorLease, TensorReadRequest, WeightStoreDiagnostics};
    use std::path::Path;
    use std::sync::Mutex;

    struct Catalog;
    struct Lease;

    #[test]
    fn independent_member_row_and_column_selections_terminate_and_preserve_values() {
        struct BankCatalog;
        impl RecipeCatalog for BankCatalog {
            fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
                Ok(TensorMetadata {
                    name: key.into(),
                    logical_shape: vec![2, 3, 4],
                    physical_shape: vec![2, 3, 4],
                    stored_dtype: StoredDtype::F32,
                    encoded_byte_len: 96,
                    backing_shard: None,
                })
            }
        }
        fn select(
            shape: Vec<usize>,
            values: Vec<usize>,
            selection: &TensorSelection,
        ) -> (Vec<usize>, Vec<usize>) {
            match selection {
                TensorSelection::Full => (shape, values),
                TensorSelection::Range { axis, start, end } => {
                    let stride: usize = shape[axis + 1..].iter().product();
                    let output = values
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, value)| {
                            let coordinate = index / stride % shape[*axis];
                            (*start <= coordinate && coordinate < *end).then_some(value)
                        })
                        .collect();
                    let mut shape = shape;
                    shape[*axis] = end - start;
                    (shape, output)
                }
                TensorSelection::Contiguous {
                    offset_elements,
                    shape,
                } => {
                    let count: usize = shape.iter().product();
                    (
                        shape.clone(),
                        values[*offset_elements..*offset_elements + count].to_vec(),
                    )
                }
                _ => panic!("unexpected selection in range fixture"),
            }
        }
        fn evaluate(
            recipe: &DerivedWeightRecipe,
            source_elements: &mut usize,
        ) -> (Vec<usize>, Vec<usize>) {
            match recipe {
                DerivedWeightRecipe::Source { selection, .. } => {
                    let result = select(vec![2, 3, 4], (0..24).collect(), selection);
                    *source_elements += result.1.len();
                    result
                }
                DerivedWeightRecipe::Select { input, selection } => {
                    let (shape, values) = evaluate(input, source_elements);
                    select(shape, values, selection)
                }
                _ => panic!("unexpected operation in range fixture"),
            }
        }
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut recipe = DerivedWeightRecipe::source("bank", TensorSelection::Full);
            for axis in order {
                recipe = recipe
                    .select_bounded(
                        &BankCatalog,
                        TensorSelection::Range {
                            axis,
                            start: 1,
                            end: if axis == 0 { 2 } else { 3 },
                        },
                    )
                    .unwrap();
            }
            let mut source_elements = 0;
            assert_eq!(
                evaluate(&recipe, &mut source_elements),
                (vec![1, 2, 2], vec![17, 18, 21, 22]),
                "{order:?}"
            );
            assert!(
                source_elements <= 8,
                "row tiles must bound source reads: {order:?}: {recipe:?}"
            );
            let mut scalar = DerivedWeightRecipe::source("bank", TensorSelection::Full);
            for axis in order {
                scalar = scalar
                    .select_bounded(
                        &BankCatalog,
                        TensorSelection::Range {
                            axis,
                            start: 1,
                            end: 2,
                        },
                    )
                    .unwrap();
            }
            let mut source_elements = 0;
            assert_eq!(
                evaluate(&scalar, &mut source_elements),
                (vec![1, 1, 1], vec![17]),
                "{order:?}"
            );
            assert_eq!(
                source_elements, 1,
                "a contiguous scalar must read only its source value: {order:?}: {scalar:?}"
            );
        }
        let columns = DerivedWeightRecipe::source(
            "bank",
            TensorSelection::Range {
                axis: 2,
                start: 1,
                end: 3,
            },
        );
        let tile = columns
            .select_bounded_matrix_rows(&BankCatalog, 1, 1, 3)
            .unwrap();
        let mut source_elements = 0;
        assert_eq!(
            evaluate(&tile, &mut source_elements),
            (vec![1, 2, 2], vec![17, 18, 21, 22])
        );
        assert!(source_elements <= 8);
    }

    #[test]
    fn distinct_recipe_validators_run_once_even_on_failure_and_under_concurrency() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Supported;
        struct Unsupported;
        let cache = RecipeInferenceCache::default();
        let recipe = DerivedWeightRecipe::source("weight", TensorSelection::Full);
        let calls = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    cache
                        .validate::<Supported>(&recipe, || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(
                        cache
                            .validate::<Unsupported>(&recipe, || {
                                calls.fetch_add(1, Ordering::Relaxed);
                                Err("unsupported encoding".into())
                            })
                            .unwrap_err(),
                        "unsupported encoding"
                    );
                });
            }
        });
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn immutable_recipe_inference_is_shared_across_subtrees_and_concurrent_calls() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CountingCatalog {
            cache: RecipeInferenceCache,
            calls: AtomicUsize,
        }
        impl RecipeCatalog for CountingCatalog {
            fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
                Some(&self.cache)
            }
            fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                if key == "missing" {
                    return Err(StoreError::UnknownTensor { key: key.into() });
                }
                Ok(TensorMetadata {
                    name: key.into(),
                    logical_shape: vec![2, 2],
                    physical_shape: vec![2, 2],
                    stored_dtype: StoredDtype::F32,
                    encoded_byte_len: 16,
                    backing_shard: None,
                })
            }
        }
        let catalog = CountingCatalog {
            cache: Default::default(),
            calls: AtomicUsize::new(0),
        };
        let stack = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: (0..64)
                .map(|i| DerivedWeightRecipe::source(format!("expert.{i}"), TensorSelection::Full))
                .collect(),
        };
        let recipe = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![stack.clone(), stack.clone()],
        };
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    assert_eq!(recipe.infer(&catalog).unwrap().shape, [128, 2, 2]);
                    assert_eq!(stack.infer(&catalog).unwrap().shape, [64, 2, 2]);
                    if let DerivedWeightRecipe::Stack { inputs, .. } = &stack {
                        for input in inputs {
                            input.infer(&catalog).unwrap();
                        }
                    }
                    assert!(
                        DerivedWeightRecipe::source("missing", TensorSelection::Full)
                            .infer(&catalog)
                            .is_err()
                    );
                });
            }
        });
        assert_eq!(catalog.calls.load(Ordering::Relaxed), 65);
    }

    #[test]
    fn borrowed_recipe_results_reuse_the_same_cached_storage_outside_its_lock() {
        struct Cached {
            cache: RecipeInferenceCache,
            calls: std::cell::Cell<usize>,
        }
        impl RecipeCatalog for Cached {
            fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
                Some(&self.cache)
            }
            fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
                self.calls.set(self.calls.get() + 1);
                Ok(TensorMetadata {
                    name: key.into(),
                    logical_shape: vec![2, 3],
                    physical_shape: vec![2, 3],
                    stored_dtype: StoredDtype::F32,
                    encoded_byte_len: 24,
                    backing_shard: None,
                })
            }
        }
        let catalog = Cached {
            cache: RecipeInferenceCache::default(),
            calls: std::cell::Cell::new(0),
        };
        let recipe = DerivedWeightRecipe::Transpose {
            input: Box::new(DerivedWeightRecipe::source("x", TensorSelection::Full)),
            axes: vec![1, 0],
        };
        recipe
            .with_inferred(&catalog, |first| {
                assert_eq!(first.shape(), &[3, 2]);
                // The second visit would deadlock if the cache mutex were retained.
                recipe
                    .with_inferred(&catalog, |second| {
                        assert!(std::ptr::eq(first, second));
                        assert_eq!(first.shape().as_ptr(), second.shape().as_ptr());
                        assert_eq!(second.borrowed().shape().collect::<Vec<_>>(), [3, 2]);
                        assert_eq!(second.borrowed().byte_len(), 24);
                    })
                    .unwrap();
            })
            .unwrap();
        assert_eq!(catalog.calls.get(), 1);
        assert_eq!(recipe.infer(&catalog).unwrap().shape(), &[3, 2]);
        assert_eq!(catalog.calls.get(), 1);
    }

    #[test]
    fn borrowed_direct_recipe_selections_preserve_validation_and_owned_fallback() {
        struct Source {
            metadata: TensorMetadata,
            loan: bool,
            calls: std::sync::atomic::AtomicUsize,
        }
        impl CheckpointSource for Source {
            fn source_metadata_borrowed(&self, _: &str) -> crate::store::SourceMetadataLoan<'_> {
                if self.loan {
                    Ok(&self.metadata)
                } else {
                    Err(crate::store::SourceMetadataBorrowError::Unavailable)
                }
            }
            fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
                assert!(
                    !self.loan,
                    "prepared successful loan must not clone the catalog"
                );
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(self.metadata.clone())
            }
            fn source_keys(&self) -> Vec<String> {
                panic!("keys are not part of inference")
            }
            fn acquire_lease(
                &self,
                _: TensorReadRequest,
            ) -> Result<crate::store::CheckpointLease, StoreError> {
                panic!("inference must not acquire payloads")
            }
            fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
                panic!("no diagnostics")
            }
        }
        let mut source = Source {
            metadata: TensorMetadata {
                name: "x".into(),
                logical_shape: vec![2, 6],
                physical_shape: vec![2, 6],
                stored_dtype: StoredDtype::F4,
                encoded_byte_len: 6,
                backing_shard: None,
            },
            loan: true,
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let cases = [
            (TensorSelection::Full, vec![2, 6], 6),
            (
                TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 5,
                },
                vec![2, 4],
                4,
            ),
            (
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![1, 0, 1],
                },
                vec![3, 6],
                9,
            ),
            (
                TensorSelection::Contiguous {
                    offset_elements: 2,
                    shape: vec![5],
                },
                vec![5],
                3,
            ),
        ];
        for (selection, expected, bytes) in &cases {
            DerivedWeightRecipe::with_source_metadata("x", selection, &source, |view| {
                assert_eq!(view.shape().collect::<Vec<_>>(), *expected);
                assert_eq!(view.byte_len(), *bytes);
                assert_eq!(view.dtype(), &RecipeDtype::F4);
            })
            .unwrap();
        }
        let invalid = [
            TensorSelection::Range {
                axis: 2,
                start: 0,
                end: 1,
            },
            TensorSelection::Range {
                axis: 1,
                start: 2,
                end: 7,
            },
            TensorSelection::Indices {
                axis: 0,
                indices: vec![2],
            },
            TensorSelection::Contiguous {
                offset_elements: 8,
                shape: vec![5],
            },
        ];
        for selection in &invalid {
            let borrowed =
                DerivedWeightRecipe::with_source_metadata("x", selection, &source, |_| {
                    panic!("invalid loan")
                });
            source.loan = false;
            let owned = DerivedWeightRecipe::source("x", selection.clone())
                .infer(&source as &dyn CheckpointSource);
            assert_eq!(
                borrowed.unwrap_err().to_string(),
                owned.unwrap_err().to_string()
            );
            source.loan = true;
        }
        source.loan = false;
        let before = source.calls.load(std::sync::atomic::Ordering::Relaxed);
        for (selection, expected, bytes) in &cases {
            DerivedWeightRecipe::with_source_metadata("x", selection, &source, |view| {
                assert_eq!(view.shape().collect::<Vec<_>>(), *expected);
                assert_eq!(view.byte_len(), *bytes);
            })
            .unwrap();
        }
        assert_eq!(
            source.calls.load(std::sync::atomic::Ordering::Relaxed),
            before + cases.len()
        );
    }

    #[test]
    fn stacked_member_selection_reads_metadata_linearly_and_preserves_exact_recipes() {
        struct CountingCatalog(std::cell::Cell<usize>);
        impl RecipeCatalog for CountingCatalog {
            fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
                self.0.set(self.0.get() + 1);
                Catalog.metadata(key)
            }
        }
        for count in [16, 128, 256] {
            let catalog = CountingCatalog(std::cell::Cell::new(0));
            let recipe = DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: (0..count)
                    .map(|member| DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: ["left", "right"]
                            .into_iter()
                            .map(|key| {
                                DerivedWeightRecipe::source(
                                    key,
                                    TensorSelection::Range {
                                        axis: 0,
                                        start: member % 2,
                                        end: member % 2 + 1,
                                    },
                                )
                            })
                            .collect(),
                    })
                    .collect(),
            };
            let members = recipe.select_bounded_members(&catalog).unwrap();
            assert!(
                catalog.0.get() <= count * 4,
                "metadata visits must scale with source leaves, got {} for {count} members",
                catalog.0.get()
            );
            assert_eq!(members.len(), count);
            for (member, actual) in members.iter().enumerate() {
                let expected = recipe
                    .select_bounded(
                        &catalog,
                        TensorSelection::Range {
                            axis: 0,
                            start: member,
                            end: member + 1,
                        },
                    )
                    .unwrap();
                assert_eq!(*actual, expected);
            }
        }
        let catalog = CountingCatalog(std::cell::Cell::new(0));
        let malformed = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source("left", TensorSelection::Full),
                DerivedWeightRecipe::source(
                    "right",
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                ),
            ],
        };
        assert!(matches!(
            malformed.select_bounded_members(&catalog),
            Err(RecipeError::ShapeMismatch)
        ));
    }

    #[derive(Default)]
    struct BoundedCatalog {
        requests: Mutex<Vec<(String, TensorSelection)>>,
    }

    impl RecipeCatalog for BoundedCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            Catalog.metadata(key)
        }
    }

    impl BoundedRecipeSource for BoundedCatalog {
        fn verify_bounded_source(
            &self,
            key: &str,
            selection: TensorSelection,
        ) -> Result<(), StoreError> {
            self.requests
                .lock()
                .unwrap()
                .push((key.to_owned(), selection));
            Ok(())
        }
    }

    impl EncodedTensorLease for Lease {
        fn metadata(&self) -> &TensorMetadata {
            unreachable!()
        }
        fn selection(&self) -> &TensorSelection {
            unreachable!()
        }
        fn output_shape(&self) -> &[usize] {
            unreachable!()
        }
        fn bounded_read_proof(&self) -> &crate::store::BoundedReadProof {
            unreachable!()
        }
        fn backing_path(&self) -> Option<&Path> {
            None
        }
        fn encoded_bytes(&self) -> Option<&[u8]> {
            None
        }
    }

    impl WeightStore for Catalog {
        type Lease = Lease;

        fn keys(&self) -> Vec<String> {
            vec!["left".into(), "right".into()]
        }
        fn metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            if !self.keys().iter().any(|candidate| candidate == key) {
                return Err(StoreError::UnknownTensor { key: key.into() });
            }
            Ok(TensorMetadata {
                name: key.into(),
                logical_shape: vec![2, 3],
                physical_shape: vec![2, 3],
                stored_dtype: StoredDtype::F16,
                encoded_byte_len: 12,
                backing_shard: None,
            })
        }
        fn acquire(&self, _: TensorReadRequest) -> Result<Self::Lease, StoreError> {
            unreachable!()
        }
        fn diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            unreachable!()
        }
    }

    #[test]
    fn nested_recipe_inference_is_backend_independent() {
        let recipe = DerivedWeightRecipe::Transpose {
            input: Box::new(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source("left", TensorSelection::Full),
                    DerivedWeightRecipe::source(
                        "right",
                        TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 1,
                        },
                    ),
                ],
            }),
            axes: vec![1, 0],
        };
        let metadata = recipe.infer(&Catalog).unwrap();
        assert_eq!(metadata.shape(), &[3, 3]);
        assert_eq!(metadata.dtype(), &RecipeDtype::F16);
        assert_eq!(metadata.byte_len(), 18);
        assert_eq!(recipe.source_keys(), ["left", "right"]);

        let ordered = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source("right", TensorSelection::Full),
                DerivedWeightRecipe::source("left", TensorSelection::Full),
                DerivedWeightRecipe::source("right", TensorSelection::Full),
            ],
        };
        assert_eq!(ordered.source_keys(), ["left", "right"]);
        assert_eq!(ordered.source_occurrences(), ["right", "left", "right"]);
    }

    #[test]
    fn bounded_preflight_walks_exact_physical_source_selections() {
        let catalog = BoundedCatalog::default();
        let recipe = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source(
                    "left",
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                ),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source("right", TensorSelection::Full)),
                    shape: vec![2, 3],
                },
            ],
        };

        recipe.preflight_bounded(&catalog).unwrap();
        assert_eq!(
            *catalog.requests.lock().unwrap(),
            vec![
                (
                    "left".into(),
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                ),
                ("right".into(), TensorSelection::Full),
            ]
        );
    }

    #[test]
    fn bounded_selection_pushdown_is_backend_independent() {
        let recipe = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source("left", TensorSelection::Full),
                DerivedWeightRecipe::source("right", TensorSelection::Full),
            ],
        };
        let selected = recipe
            .select_bounded(
                &Catalog,
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 3,
                },
            )
            .unwrap();
        assert_eq!(selected.infer(&Catalog).unwrap().shape(), &[2, 3]);
        assert_eq!(
            selected,
            DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source(
                        "left",
                        TensorSelection::Range {
                            axis: 0,
                            start: 1,
                            end: 2,
                        },
                    ),
                    DerivedWeightRecipe::source(
                        "right",
                        TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 1,
                        },
                    ),
                ],
            }
        );
    }

    #[test]
    fn fused_members_and_companions_validate_as_one_atomic_recipe_set() {
        let split = atomic_fused_split_recipes(
            &Catalog,
            [
                FusedSplitMember {
                    source: "left".into(),
                    axis: 1,
                    outputs: vec![
                        FusedSplitOutput {
                            target: "weight.query".into(),
                            width: 1,
                        },
                        FusedSplitOutput {
                            target: "weight.key_value".into(),
                            width: 2,
                        },
                    ],
                },
                FusedSplitMember {
                    source: "right".into(),
                    axis: 1,
                    outputs: vec![
                        FusedSplitOutput {
                            target: "scale.query".into(),
                            width: 1,
                        },
                        FusedSplitOutput {
                            target: "scale.key_value".into(),
                            width: 2,
                        },
                    ],
                },
            ],
        )
        .unwrap();
        assert_eq!(split.iter().count(), 4);
        assert_eq!(
            split
                .get("weight.key_value")
                .unwrap()
                .infer(&Catalog)
                .unwrap()
                .shape(),
            &[2, 2]
        );

        let duplicate = atomic_fused_split_recipes(
            &Catalog,
            [FusedSplitMember {
                source: "left".into(),
                axis: 1,
                outputs: vec![
                    FusedSplitOutput {
                        target: "same".into(),
                        width: 1,
                    },
                    FusedSplitOutput {
                        target: "same".into(),
                        width: 2,
                    },
                ],
            }],
        );
        assert!(matches!(
            duplicate,
            Err(RecipeError::DuplicateOutput { .. })
        ));

        let mismatch = atomic_fused_split_recipes(
            &Catalog,
            [FusedSplitMember {
                source: "left".into(),
                axis: 1,
                outputs: vec![FusedSplitOutput {
                    target: "short".into(),
                    width: 2,
                }],
            }],
        );
        assert!(matches!(
            mismatch,
            Err(RecipeError::FusedSplitWidthMismatch { .. })
        ));
    }

    #[test]
    fn ordered_axis_selection_validates_value_head_layout_without_payload_reads() {
        let recipe = ordered_axis_selection(&Catalog, "left", 1, vec![2, 0, 1]).unwrap();
        assert_eq!(recipe.infer(&Catalog).unwrap().shape(), &[2, 3]);
        assert!(ordered_axis_selection(&Catalog, "left", 1, vec![3]).is_err());
    }

    struct FamilyCatalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for FamilyCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn family_catalog(entries: &[(&str, &[usize], StoredDtype)]) -> FamilyCatalog {
        FamilyCatalog(
            entries
                .iter()
                .map(|(name, shape, dtype)| {
                    (
                        (*name).to_owned(),
                        TensorMetadata {
                            name: (*name).to_owned(),
                            logical_shape: shape.to_vec(),
                            physical_shape: shape.to_vec(),
                            stored_dtype: dtype.clone(),
                            encoded_byte_len: 1,
                            backing_shard: Some("synthetic.safetensors".into()),
                        },
                    )
                })
                .collect(),
        )
    }

    fn member(target: &str, source: &str) -> MatrixRecipeMember {
        MatrixRecipeMember::new(
            target,
            DerivedWeightRecipe::source(source, TensorSelection::Full),
        )
    }

    #[test]
    fn dense_matrix_family_selects_leading_rows_atomically() {
        let catalog = family_catalog(&[("dense.weight", &[4, 6], StoredDtype::F16)]);
        let family = AtomicMatrixRecipeFamily::new(
            &catalog,
            member("model.weight", "dense.weight"),
            None,
            None,
        )
        .unwrap();
        let selected = family
            .select_leading_axis(
                &catalog,
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 3,
                },
            )
            .unwrap();
        assert_eq!(
            selected.weight().recipe.infer(&catalog).unwrap().shape(),
            &[2, 6]
        );
        let published = selected.publish(&catalog, []).unwrap();
        assert_eq!(published.iter().count(), 1);
        assert_eq!(published.aliases().count(), 0);
        assert_eq!(
            published.get("model.weight").unwrap(),
            &DerivedWeightRecipe::source(
                "dense.weight",
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 3,
                }
            )
        );
    }

    #[test]
    fn affine_matrix_family_selects_weight_scales_and_biases_coherently() {
        let catalog = family_catalog(&[
            ("affine.weight", &[4, 3], StoredDtype::U32),
            ("affine.scales", &[4, 2], StoredDtype::F16),
            ("affine.biases", &[4, 2], StoredDtype::F16),
        ]);
        let family = AtomicMatrixRecipeFamily::new(
            &catalog,
            member("model.weight", "affine.weight"),
            Some(member("model.scales", "affine.scales")),
            Some(member("model.biases", "affine.biases")),
        )
        .unwrap();
        let selected = family
            .select_leading_axis(
                &catalog,
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![3, 1],
                },
            )
            .unwrap();
        let published = selected.publish(&catalog, []).unwrap();
        assert_eq!(published.iter().count(), 3);
        assert_eq!(
            published
                .get("model.weight")
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[2, 3]
        );
        for companion in ["model.scales", "model.biases"] {
            assert_eq!(
                published
                    .get(companion)
                    .unwrap()
                    .infer(&catalog)
                    .unwrap()
                    .shape(),
                &[2, 2]
            );
        }
    }

    #[test]
    fn mxfp4_matrix_family_preserves_scale_companion_without_biases() {
        let catalog = family_catalog(&[
            ("mxfp4.weight", &[4, 32], StoredDtype::F4),
            ("mxfp4.scales", &[4, 1], StoredDtype::F8E8M0),
        ]);
        let family = AtomicMatrixRecipeFamily::new(
            &catalog,
            member("model.weight", "mxfp4.weight"),
            Some(member("model.scales", "mxfp4.scales")),
            None,
        )
        .unwrap();
        let selected = family
            .select_leading_axis(
                &catalog,
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )
            .unwrap();
        assert_eq!(
            selected.weight().recipe.infer(&catalog).unwrap().dtype(),
            &RecipeDtype::F4
        );
        assert_eq!(
            selected
                .scales()
                .unwrap()
                .recipe
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[1, 1]
        );
        assert!(selected.biases().is_none());
    }

    #[test]
    fn matrix_family_rejects_malformed_companions_before_publication() {
        let catalog = family_catalog(&[
            ("weight", &[4, 8], StoredDtype::U32),
            ("bad.scales", &[3, 2], StoredDtype::F16),
            ("scales", &[4, 2], StoredDtype::F16),
            ("bad.biases", &[4, 1], StoredDtype::F16),
        ]);
        assert!(matches!(
            AtomicMatrixRecipeFamily::new(
                &catalog,
                member("model.weight", "weight"),
                Some(member("model.scales", "bad.scales")),
                None,
            ),
            Err(RecipeError::MatrixCompanionGeometry { .. })
        ));
        assert!(matches!(
            AtomicMatrixRecipeFamily::new(
                &catalog,
                member("model.weight", "weight"),
                None,
                Some(member("model.biases", "bad.biases")),
            ),
            Err(RecipeError::MatrixBiasWithoutScales)
        ));
        assert!(matches!(
            AtomicMatrixRecipeFamily::new(
                &catalog,
                member("model.weight", "weight"),
                Some(member("model.scales", "scales")),
                Some(member("model.biases", "bad.biases")),
            ),
            Err(RecipeError::MatrixScaleBiasGeometry { .. })
        ));

        let valid = AtomicMatrixRecipeFamily::new(
            &catalog,
            member("model.weight", "weight"),
            Some(member("model.scales", "scales")),
            None,
        )
        .unwrap();
        assert!(matches!(
            valid.select_leading_axis(
                &catalog,
                TensorSelection::Range {
                    axis: 1,
                    start: 0,
                    end: 1,
                }
            ),
            Err(RecipeError::MatrixFamilySelectionAxis { axis: 1 })
        ));
    }

    #[test]
    fn aliases_resolve_to_one_canonical_owner_without_recipe_duplication() {
        let catalog = family_catalog(&[("shared", &[2, 3], StoredDtype::F16)]);
        let family = AtomicMatrixRecipeFamily::new(
            &catalog,
            member("canonical.weight", "shared"),
            None,
            None,
        )
        .unwrap();
        let published = family
            .publish(
                &catalog,
                [
                    RecipeAlias::new("slice.1.weight", "canonical.weight"),
                    RecipeAlias::new("slice.2.weight", "slice.1.weight"),
                ],
            )
            .unwrap();
        assert_eq!(
            published.aliases().collect::<Vec<_>>(),
            vec![
                ("slice.1.weight", "canonical.weight"),
                ("slice.2.weight", "canonical.weight"),
            ]
        );
        let (_, canonical) = published.get_resolved("canonical.weight").unwrap();
        let (owner, aliased) = published.get_resolved("slice.2.weight").unwrap();
        assert_eq!(owner, "canonical.weight");
        assert!(std::ptr::eq(canonical, aliased));
        let (outputs, aliases) = published.into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(aliases.len(), 2);
    }

    #[test]
    fn alias_validation_rejects_collision_cycle_and_unknown_destination_atomically() {
        let catalog = family_catalog(&[("shared", &[2, 3], StoredDtype::F16)]);
        let outputs = || {
            [(
                "canonical.weight".to_owned(),
                DerivedWeightRecipe::source("shared", TensorSelection::Full),
            )]
        };
        assert!(matches!(
            AtomicRecipeSet::new_with_aliases(
                &catalog,
                outputs(),
                [RecipeAlias::new("canonical.weight", "canonical.weight")],
            ),
            Err(RecipeError::AliasOutputCollision { .. })
        ));
        assert!(matches!(
            AtomicRecipeSet::new_with_aliases(
                &catalog,
                outputs(),
                [
                    RecipeAlias::new("first", "second"),
                    RecipeAlias::new("second", "first"),
                ],
            ),
            Err(RecipeError::AliasCycle { .. })
        ));
        assert!(matches!(
            AtomicRecipeSet::new_with_aliases(
                &catalog,
                outputs(),
                [RecipeAlias::new("orphan", "missing.weight")],
            ),
            Err(RecipeError::InvalidAliasDestination { .. })
        ));
        assert!(matches!(
            AtomicRecipeSet::new_with_aliases(
                &catalog,
                outputs(),
                [
                    RecipeAlias::new("duplicate", "canonical.weight"),
                    RecipeAlias::new("duplicate", "canonical.weight"),
                ],
            ),
            Err(RecipeError::DuplicateAlias { .. })
        ));
    }
}

#[cfg(test)]
mod prepared_source_visits {
    use super::*;
    #[test]
    fn actual_occurrences_share_order_selection_join_events_and_early_failure() {
        struct Visitor {
            rows: Vec<(String, TensorSelection, usize)>,
            multiplicity: usize,
            stop: bool,
        }
        impl RecipeSourceVisitor for Visitor {
            type Error = &'static str;
            fn source(
                &mut self,
                key: &str,
                selection: &TensorSelection,
            ) -> Result<(), Self::Error> {
                if self.stop && key == "last" {
                    return Err("actual last refusal");
                }
                self.rows
                    .push((key.into(), selection.clone(), self.multiplicity));
                Ok(())
            }
            fn enter_join(&mut self) -> Result<(), Self::Error> {
                self.multiplicity *= 2;
                Ok(())
            }
            fn leave_join(&mut self) {
                self.multiplicity /= 2;
            }
        }
        let selected = TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        };
        let recipe = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source("same", selected.clone()),
                DerivedWeightRecipe::NegLog {
                    input: Box::new(DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![
                            DerivedWeightRecipe::source("same", TensorSelection::Full),
                            DerivedWeightRecipe::source("last", TensorSelection::Full),
                        ],
                    }),
                },
            ],
        };
        let mut visitor = Visitor {
            rows: Vec::new(),
            multiplicity: 4,
            stop: false,
        };
        recipe.visit_sources(&mut visitor).unwrap();
        assert_eq!(recipe.source_occurrences(), ["same", "same", "last"]);
        assert_eq!(
            visitor.rows,
            [
                ("same".into(), selected, 8),
                ("same".into(), TensorSelection::Full, 16),
                ("last".into(), TensorSelection::Full, 16)
            ]
        );
        assert_eq!(visitor.multiplicity, 4);
        visitor.rows.clear();
        visitor.stop = true;
        assert_eq!(
            recipe.visit_sources(&mut visitor),
            Err("actual last refusal")
        );
        assert_eq!(visitor.rows.len(), 2);
    }
}
