//! Neutral lowering of logical parameter placement into bounded checkpoint selections.

use std::collections::{BTreeMap, BTreeSet};

use eredu_checkpoint::{
    recipe::RecipeError,
    store::{CheckpointSource, StoreError, TensorSelection},
};

use crate::{
    LocalModelLayout, LocalTensorLayout, ResidencyDeclarationError, TensorPlacement, WeightBinding,
    WeightBindingSelectionError,
};

/// Validated world identity used by logical checkpoint placement.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PlacementRank {
    world_size: usize,
    global_rank: usize,
}

impl PlacementRank {
    /// Creates one rank in a non-empty world.
    pub fn new(world_size: usize, global_rank: usize) -> Result<Self, PlacementPlanError> {
        if world_size == 0 || global_rank >= world_size {
            return Err(PlacementPlanError::InvalidRank {
                world_size,
                global_rank,
            });
        }
        Ok(Self {
            world_size,
            global_rank,
        })
    }

    /// Returns the process count.
    pub const fn world_size(self) -> usize {
        self.world_size
    }

    /// Returns the selected process rank.
    pub const fn global_rank(self) -> usize {
        self.global_rank
    }
}

/// A validated contiguous slice of a source tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorSlice {
    axis: usize,
    start: usize,
    end: usize,
    index: usize,
    parts: usize,
}

impl TensorSlice {
    /// Validates and calculates an equal contiguous tensor slice.
    pub fn for_shape(
        shape: &[usize],
        axis: usize,
        index: usize,
        parts: usize,
    ) -> Result<Self, PlacementPlanError> {
        let dimension = *shape.get(axis).ok_or(PlacementPlanError::AxisOutOfBounds {
            axis,
            rank: shape.len(),
        })?;
        if parts == 0 || index >= parts || dimension == 0 || !dimension.is_multiple_of(parts) {
            return Err(PlacementPlanError::InvalidShard {
                axis,
                index,
                parts,
                dimension,
            });
        }
        let width = dimension / parts;
        let start = index
            .checked_mul(width)
            .ok_or(PlacementPlanError::ArithmeticOverflow)?;
        Ok(Self {
            axis,
            start,
            end: start + width,
            index,
            parts,
        })
    }

    /// Returns the source axis.
    pub const fn axis(&self) -> usize {
        self.axis
    }
    /// Returns the inclusive source offset.
    pub const fn start(&self) -> usize {
        self.start
    }
    /// Returns the exclusive source offset.
    pub const fn end(&self) -> usize {
        self.end
    }
    /// Returns the shard index.
    pub const fn index(&self) -> usize {
        self.index
    }
    /// Returns the shard count.
    pub const fn parts(&self) -> usize {
        self.parts
    }
    /// Returns the resulting local shape.
    pub fn local_shape(&self, source_shape: &[usize]) -> Vec<usize> {
        let mut shape = source_shape.to_vec();
        shape[self.axis] = self.end - self.start;
        shape
    }
}

#[derive(Debug, Clone)]
struct TensorPlan {
    placement: TensorPlacement,
    expected_source_shape: Option<Vec<usize>>,
}

/// Exact logical checkpoint placement for one rank.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    rank: PlacementRank,
    tensors: BTreeMap<String, TensorPlan>,
    default: Option<TensorPlacement>,
}

impl PlacementPlan {
    /// Creates a strict plan in which every checkpoint source must be named.
    pub const fn new(rank: PlacementRank) -> Self {
        Self {
            rank,
            tensors: BTreeMap::new(),
            default: None,
        }
    }

    /// Creates a plan that replicates every checkpoint source.
    pub fn replicated(rank: PlacementRank) -> Self {
        Self::new(rank).with_default(TensorPlacement::Replicated)
    }

    /// Returns the selected logical rank.
    pub const fn rank(&self) -> PlacementRank {
        self.rank
    }

    /// Sets the placement for otherwise unnamed checkpoint sources.
    pub fn with_default(mut self, placement: TensorPlacement) -> Self {
        self.default = Some(placement);
        self
    }

    /// Adds or replaces one exact source placement.
    pub fn insert(&mut self, source: impl Into<String>, placement: TensorPlacement) {
        self.tensors.insert(
            source.into(),
            TensorPlan {
                placement,
                expected_source_shape: None,
            },
        );
    }

    /// Adds one source placement with an exact pre-selection shape.
    pub fn insert_expected(
        &mut self,
        source: impl Into<String>,
        expected_source_shape: impl Into<Vec<usize>>,
        placement: TensorPlacement,
    ) -> Result<(), PlacementPlanError> {
        let shape = expected_source_shape.into();
        validate_tensor_placement(&placement, &shape, self.rank)?;
        self.tensors.insert(
            source.into(),
            TensorPlan {
                placement,
                expected_source_shape: Some(shape),
            },
        );
        Ok(())
    }

    /// Adds a packed weight and its scale and optional bias companions together.
    pub fn insert_quantized_companions(
        &mut self,
        prefix: &str,
        placement: TensorPlacement,
        has_biases: bool,
    ) {
        self.insert(format!("{prefix}.weight"), placement.clone());
        self.insert(format!("{prefix}.scales"), placement.clone());
        if has_biases {
            self.insert(format!("{prefix}.biases"), placement);
        }
    }

    /// Returns an explicit placement by exact source name.
    pub fn placement(&self, source: &str) -> Option<&TensorPlacement> {
        self.tensors.get(source).map(|plan| &plan.placement)
    }

    /// Validates all geometry available before checkpoint access.
    pub fn validate(&self) -> Result<(), PlacementPlanError> {
        for tensor in self.tensors.values() {
            validate_tensor_plan(tensor, self.rank)?;
        }
        if let Some(default) = &self.default {
            validate_tensor_plan(
                &TensorPlan {
                    placement: default.clone(),
                    expected_source_shape: None,
                },
                self.rank,
            )?;
        }
        Ok(())
    }

    /// Returns whether a named source can require materialization on this rank.
    pub fn potentially_local(&self, source: &str) -> Result<bool, PlacementPlanError> {
        let plan = self.source_plan(source)?;
        Ok(!matches!(plan.placement, TensorPlacement::Omit)
            && !matches!(plan.placement, TensorPlacement::Rank { rank } if rank != self.rank.global_rank))
    }

    /// Resolves one named source against its admitted logical shape.
    pub fn resolve(
        &self,
        source: &str,
        shape: &[usize],
    ) -> Result<ResolvedTensorPlacement, PlacementPlanError> {
        let plan = self.source_plan(source)?;
        if plan
            .expected_source_shape
            .as_ref()
            .is_some_and(|expected| expected != shape)
        {
            return Err(PlacementPlanError::SourceShapeMismatch {
                checkpoint_source: source.to_owned(),
                expected: plan.expected_source_shape.clone().unwrap(),
                actual: shape.to_vec(),
            });
        }
        validate_tensor_placement(&plan.placement, shape, self.rank)?;
        Ok(match &plan.placement {
            TensorPlacement::Replicated | TensorPlacement::Local => {
                ResolvedTensorPlacement::Materialize
            }
            TensorPlacement::Omit => ResolvedTensorPlacement::Omit,
            TensorPlacement::Rank { rank } if *rank == self.rank.global_rank => {
                ResolvedTensorPlacement::Materialize
            }
            TensorPlacement::Rank { .. } => ResolvedTensorPlacement::Omit,
            TensorPlacement::Shard { axis, index, parts } => {
                let slice = TensorSlice::for_shape(shape, *axis, *index, *parts)?;
                ResolvedTensorPlacement::Selection(TensorSelection::Range {
                    axis: slice.axis,
                    start: slice.start,
                    end: slice.end,
                })
            }
            TensorPlacement::Range { axis, start, end } => {
                ResolvedTensorPlacement::Selection(TensorSelection::Range {
                    axis: *axis,
                    start: *start,
                    end: *end,
                })
            }
            TensorPlacement::Indices { axis, indices } => {
                ResolvedTensorPlacement::Selection(TensorSelection::Indices {
                    axis: *axis,
                    indices: indices.clone(),
                })
            }
        })
    }

    /// Verifies exact locally required and unexpected source coverage.
    pub fn validate_loaded_sources(
        &self,
        loaded: &BTreeSet<String>,
        mut unexpected: Vec<String>,
    ) -> Result<(), PlacementPlanError> {
        let mut missing = self
            .tensors
            .iter()
            .filter_map(|(source, plan)| {
                let required = !matches!(plan.placement, TensorPlacement::Omit)
                    && !matches!(plan.placement, TensorPlacement::Rank { rank } if rank != self.rank.global_rank);
                (required && !loaded.contains(source)).then_some(source.clone())
            })
            .collect::<Vec<_>>();
        missing.sort();
        unexpected.sort();
        unexpected.dedup();
        if missing.is_empty() && unexpected.is_empty() {
            Ok(())
        } else {
            Err(PlacementPlanError::Coverage {
                missing,
                unexpected,
            })
        }
    }

    fn source_plan(&self, source: &str) -> Result<TensorPlan, PlacementPlanError> {
        self.tensors
            .get(source)
            .cloned()
            .or_else(|| {
                self.default.as_ref().map(|placement| TensorPlan {
                    placement: placement.clone(),
                    expected_source_shape: None,
                })
            })
            .ok_or_else(|| PlacementPlanError::UnexpectedSource {
                checkpoint_source: source.to_owned(),
            })
    }
}

/// Rank-local result of resolving one logical placement.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ResolvedTensorPlacement {
    /// Materialize the complete source.
    Materialize,
    /// Do not access the source on this rank.
    Omit,
    /// Materialize one exact bounded source selection.
    Selection(TensorSelection),
}

fn validate_tensor_plan(plan: &TensorPlan, rank: PlacementRank) -> Result<(), PlacementPlanError> {
    match &plan.placement {
        TensorPlacement::Rank { rank: owner } if *owner >= rank.world_size => {
            Err(PlacementPlanError::OwnerOutOfBounds {
                owner: *owner,
                world_size: rank.world_size,
            })
        }
        TensorPlacement::Shard { axis, index, parts } if *parts == 0 || *index >= *parts => {
            Err(PlacementPlanError::InvalidShard {
                axis: *axis,
                index: *index,
                parts: *parts,
                dimension: 0,
            })
        }
        TensorPlacement::Range { axis, start, end } if start >= end => {
            Err(PlacementPlanError::InvalidRange {
                axis: *axis,
                start: *start,
                end: *end,
                dimension: None,
            })
        }
        TensorPlacement::Indices { axis, indices }
            if indices.is_empty()
                || indices.iter().collect::<BTreeSet<_>>().len() != indices.len() =>
        {
            Err(PlacementPlanError::InvalidIndices { axis: *axis })
        }
        placement => {
            if let Some(shape) = &plan.expected_source_shape {
                validate_tensor_placement(placement, shape, rank)?;
            }
            Ok(())
        }
    }
}

fn validate_tensor_placement(
    placement: &TensorPlacement,
    shape: &[usize],
    rank: PlacementRank,
) -> Result<(), PlacementPlanError> {
    match placement {
        TensorPlacement::Rank { rank: owner } if *owner >= rank.world_size => {
            Err(PlacementPlanError::OwnerOutOfBounds {
                owner: *owner,
                world_size: rank.world_size,
            })
        }
        TensorPlacement::Shard { axis, index, parts } => {
            TensorSlice::for_shape(shape, *axis, *index, *parts).map(|_| ())
        }
        TensorPlacement::Range { axis, start, end } => {
            let dimension = *shape
                .get(*axis)
                .ok_or(PlacementPlanError::AxisOutOfBounds {
                    axis: *axis,
                    rank: shape.len(),
                })?;
            if start >= end || *end > dimension {
                Err(PlacementPlanError::InvalidRange {
                    axis: *axis,
                    start: *start,
                    end: *end,
                    dimension: Some(dimension),
                })
            } else {
                Ok(())
            }
        }
        TensorPlacement::Indices { axis, indices } => {
            let dimension = *shape
                .get(*axis)
                .ok_or(PlacementPlanError::AxisOutOfBounds {
                    axis: *axis,
                    rank: shape.len(),
                })?;
            if indices.is_empty()
                || indices.iter().collect::<BTreeSet<_>>().len() != indices.len()
                || indices.iter().any(|index| *index >= dimension)
            {
                Err(PlacementPlanError::InvalidIndices { axis: *axis })
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

/// Failure while validating or resolving a complete logical placement plan.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum PlacementPlanError {
    /// The selected world/rank pair is invalid.
    #[error("global rank {global_rank} is outside world size {world_size}")]
    InvalidRank {
        /// Process count.
        world_size: usize,
        /// Selected process rank.
        global_rank: usize,
    },
    /// A placement owner exceeds the selected world.
    #[error("owner rank {owner} is outside world size {world_size}")]
    OwnerOutOfBounds {
        /// Invalid owner.
        owner: usize,
        /// Process count.
        world_size: usize,
    },
    /// A tensor axis exceeds its rank.
    #[error("tensor axis {axis} is outside rank {rank}")]
    AxisOutOfBounds {
        /// Invalid axis.
        axis: usize,
        /// Tensor rank.
        rank: usize,
    },
    /// Equal-shard geometry is invalid.
    #[error("shard {index}/{parts} on axis {axis} is invalid for dimension {dimension}")]
    InvalidShard {
        /// Selected axis.
        axis: usize,
        /// Shard index.
        index: usize,
        /// Shard count.
        parts: usize,
        /// Source dimension.
        dimension: usize,
    },
    /// Explicit range geometry is invalid.
    #[error("range {start}..{end} on axis {axis} is invalid for dimension {dimension:?}")]
    InvalidRange {
        /// Selected axis.
        axis: usize,
        /// Inclusive offset.
        start: usize,
        /// Exclusive offset.
        end: usize,
        /// Known source dimension.
        dimension: Option<usize>,
    },
    /// Explicit indices are empty, duplicated, or out of bounds.
    #[error("index selection on axis {axis} is invalid")]
    InvalidIndices {
        /// Selected axis.
        axis: usize,
    },
    /// Slice arithmetic overflowed.
    #[error("tensor slice arithmetic overflowed")]
    ArithmeticOverflow,
    /// A strict plan did not declare one checkpoint source.
    #[error("checkpoint source {checkpoint_source:?} is unexpected")]
    UnexpectedSource {
        /// Undeclared checkpoint source.
        checkpoint_source: String,
    },
    /// Admitted and actual source shapes differ.
    #[error("checkpoint source {checkpoint_source:?} has shape {actual:?}, expected {expected:?}")]
    SourceShapeMismatch {
        /// Exact checkpoint source.
        checkpoint_source: String,
        /// Admitted shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },
    /// Local materialization did not exactly cover the plan.
    #[error("placement coverage mismatch: missing {missing:?}, unexpected {unexpected:?}")]
    Coverage {
        /// Required sources not loaded.
        missing: Vec<String>,
        /// Undeclared sources encountered.
        unexpected: Vec<String>,
    },
}

/// Applies all architecture-selected logical placements to parameter recipes.
pub fn place_weight_bindings(
    bindings: Vec<WeightBinding>,
    source: &dyn CheckpointSource,
    layout: &LocalModelLayout,
) -> Result<Vec<WeightBinding>, BindingPlacementError> {
    apply_binding_placements(bindings, source, layout, false)
}

/// Applies remaining non-member placements to addressable-bank bindings.
///
/// The bank catalog has already selected semantic axis zero, so the ordinary
/// logical lowering skips only that consumed member axis.
pub fn place_addressable_member_bindings(
    bindings: Vec<WeightBinding>,
    source: &dyn CheckpointSource,
    layout: &LocalModelLayout,
) -> Result<Vec<WeightBinding>, BindingPlacementError> {
    apply_binding_placements(bindings, source, layout, true)
}

fn apply_binding_placements(
    bindings: Vec<WeightBinding>,
    source: &dyn CheckpointSource,
    layout: &LocalModelLayout,
    skip_member_axis: bool,
) -> Result<Vec<WeightBinding>, BindingPlacementError> {
    let source_keys = source.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut output = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if binding.is_alias() {
            output.push(binding);
            continue;
        }
        let logical_target = binding
            .logical_target()
            .ok_or_else(|| BindingPlacementError::MissingLogicalTarget {
                binding: binding.name().to_owned(),
            })?
            .to_owned();
        let tensor = layout.tensor(&logical_target).ok_or_else(|| {
            BindingPlacementError::UnknownLogicalTarget {
                binding: binding.name().to_owned(),
                target: logical_target.clone(),
            }
        })?;
        let companions = binding.quantization_companions().cloned();

        if !skip_member_axis
            && binding.recipe().is_none()
            && !source_keys.contains(binding.checkpoint_key())
        {
            if !tensor.additional_placements().is_empty() {
                return Err(BindingPlacementError::CompoundPlacementRequiresRecipe {
                    binding: binding.name().to_owned(),
                });
            }
            let selection = placement_selection(tensor, tensor.placement(), tensor.global_shape())?;
            if selection == TensorSelection::Full {
                output.push(binding);
                continue;
            }
            let global = tensor
                .global_shape()
                .iter()
                .try_fold(1usize, |value, item| value.checked_mul(*item));
            let local = tensor
                .local_shape()
                .iter()
                .try_fold(1usize, |value, item| value.checked_mul(*item));
            let expected_bytes = global
                .zip(local)
                .and_then(|(global, local)| {
                    binding
                        .expected_bytes()
                        .checked_mul(local as u64)
                        .and_then(|bytes| bytes.checked_div(global as u64))
                })
                .ok_or_else(|| BindingPlacementError::ByteGeometry {
                    binding: binding.name().to_owned(),
                })?;
            let mut placed = WeightBinding::new(
                binding.name(),
                binding.checkpoint_key(),
                selection,
                expected_bytes,
            )?
            .with_logical_target(logical_target)?;
            if let Some(companions) = companions {
                placed = placed.with_quantization_companions(
                    companions.scale(),
                    companions.affine_bias().map(str::to_owned),
                )?;
            }
            output.push(placed);
            continue;
        }

        let mut recipe = binding.source_recipe();
        let mut selected = false;
        for placement in tensor
            .additional_placements()
            .iter()
            .chain(std::iter::once(tensor.placement()))
        {
            let member_axis = matches!(
                placement,
                TensorPlacement::Shard { axis: 0, .. }
                    | TensorPlacement::Range { axis: 0, .. }
                    | TensorPlacement::Indices { axis: 0, .. }
            );
            if skip_member_axis && member_axis {
                continue;
            }
            let metadata = recipe.infer(source)?;
            let selection = placement_selection(tensor, placement, metadata.shape())?;
            if selection != TensorSelection::Full {
                recipe = recipe.select_bounded(source, selection)?;
                selected = true;
            }
        }
        if !selected {
            output.push(binding);
            continue;
        }
        let expected_bytes = recipe.infer(source)?.byte_len();
        let mut placed = WeightBinding::from_recipe(binding.name(), recipe, expected_bytes)?
            .with_logical_target(logical_target)?;
        if let Some(companions) = companions {
            placed = placed.with_quantization_companions(
                companions.scale(),
                companions.affine_bias().map(str::to_owned),
            )?;
        }
        output.push(placed);
    }
    Ok(output)
}

/// Lowers one validated logical placement against exact stored geometry.
pub fn placement_selection(
    tensor: &LocalTensorLayout,
    placement: &TensorPlacement,
    stored_shape: &[usize],
) -> Result<TensorSelection, BindingPlacementError> {
    let scale_boundary = |axis: usize, boundary: usize| -> Result<usize, BindingPlacementError> {
        let semantic =
            *tensor
                .global_shape()
                .get(axis)
                .ok_or(BindingPlacementError::AxisOutOfBounds {
                    axis,
                    rank: tensor.global_shape().len(),
                })?;
        let stored = *stored_shape
            .get(axis)
            .ok_or(BindingPlacementError::AxisOutOfBounds {
                axis,
                rank: stored_shape.len(),
            })?;
        boundary
            .checked_mul(stored)
            .and_then(|value| value.checked_div(semantic))
            .filter(|scaled| *scaled * semantic == boundary * stored)
            .ok_or(BindingPlacementError::UnalignedBoundary {
                axis,
                boundary,
                semantic,
                stored,
            })
    };

    Ok(match placement {
        TensorPlacement::Replicated | TensorPlacement::Local => TensorSelection::Full,
        TensorPlacement::Shard { axis, index, parts } => {
            let stored =
                *stored_shape
                    .get(*axis)
                    .ok_or(BindingPlacementError::AxisOutOfBounds {
                        axis: *axis,
                        rank: stored_shape.len(),
                    })?;
            if *parts == 0 || *index >= *parts || !stored.is_multiple_of(*parts) {
                return Err(BindingPlacementError::InvalidShard {
                    axis: *axis,
                    index: *index,
                    parts: *parts,
                    stored,
                });
            }
            let width = stored / *parts;
            TensorSelection::Range {
                axis: *axis,
                start: index * width,
                end: (index + 1) * width,
            }
        }
        TensorPlacement::Range { axis, start, end } => TensorSelection::Range {
            axis: *axis,
            start: scale_boundary(*axis, *start)?,
            end: scale_boundary(*axis, *end)?,
        },
        TensorPlacement::Indices { axis, indices } => {
            let stored =
                *stored_shape
                    .get(*axis)
                    .ok_or(BindingPlacementError::AxisOutOfBounds {
                        axis: *axis,
                        rank: stored_shape.len(),
                    })?;
            let semantic = *tensor.global_shape().get(*axis).ok_or(
                BindingPlacementError::AxisOutOfBounds {
                    axis: *axis,
                    rank: tensor.global_shape().len(),
                },
            )?;
            if stored != semantic {
                return Err(BindingPlacementError::IndexedPackedStorage {
                    axis: *axis,
                    semantic,
                    stored,
                });
            }
            TensorSelection::Indices {
                axis: *axis,
                indices: indices.clone(),
            }
        }
        TensorPlacement::Omit | TensorPlacement::Rank { .. } => {
            return Err(BindingPlacementError::NonLocalPlacement {
                placement: placement.clone(),
            });
        }
    })
}

/// Failure while lowering logical placement into bounded source selections.
#[derive(Debug, thiserror::Error)]
pub enum BindingPlacementError {
    /// A binding omitted its architecture-logical target.
    #[error("binding {binding:?} has no logical placement target")]
    MissingLogicalTarget {
        /// Binding without a logical target.
        binding: String,
    },
    /// The logical target is absent from the selected layout.
    #[error("binding {binding:?} targets unknown layout entry {target:?}")]
    UnknownLogicalTarget {
        /// Binding being placed.
        binding: String,
        /// Missing layout target.
        target: String,
    },
    /// A source-less direct declaration cannot express compound placement.
    #[error("compound placement for {binding:?} requires an admitted checkpoint recipe")]
    CompoundPlacementRequiresRecipe {
        /// Binding requiring an admitted recipe.
        binding: String,
    },
    /// Rank-local byte calculation overflowed or was not integral.
    #[error("cannot derive rank-local byte geometry for {binding:?}")]
    ByteGeometry {
        /// Binding with invalid byte geometry.
        binding: String,
    },
    /// A placement axis exceeds a tensor rank.
    #[error("placement axis {axis} is outside tensor rank {rank}")]
    AxisOutOfBounds {
        /// Invalid axis.
        axis: usize,
        /// Available tensor rank.
        rank: usize,
    },
    /// A semantic boundary is not exactly representable in packed storage.
    #[error("semantic boundary {boundary} on axis {axis} ({semantic}) is not aligned to stored width {stored}")]
    UnalignedBoundary {
        /// Selected axis.
        axis: usize,
        /// Semantic boundary.
        boundary: usize,
        /// Complete semantic width.
        semantic: usize,
        /// Complete stored width.
        stored: usize,
    },
    /// Equal-shard geometry is invalid.
    #[error("shard {index}/{parts} on axis {axis} is invalid for stored width {stored}")]
    InvalidShard {
        /// Selected axis.
        axis: usize,
        /// Selected shard index.
        index: usize,
        /// Requested shard count.
        parts: usize,
        /// Complete stored width.
        stored: usize,
    },
    /// Indexed semantic selection cannot address differently packed storage.
    #[error("indexed axis {axis} has semantic width {semantic} but stored width {stored}")]
    IndexedPackedStorage {
        /// Selected axis.
        axis: usize,
        /// Complete semantic width.
        semantic: usize,
        /// Complete stored width.
        stored: usize,
    },
    /// Execution-unit binding selected an omit or remote-rank placement.
    #[error("execution-unit binding has non-local placement {placement:?}")]
    NonLocalPlacement {
        /// Rejected non-local placement.
        placement: TensorPlacement,
    },
    /// Checkpoint access failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// Recipe inference failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Lowered residency declaration is invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
    /// Bounded source selection failed.
    #[error(transparent)]
    Selection(#[from] WeightBindingSelectionError),
}
