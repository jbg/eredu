//! Ordered source occurrences for the shared encoded recipe compiler.
use super::{DerivedWeightRecipe, RecipeError, TensorSelection};
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

/// Borrows a recipe and sizes its ordered source-key destinations before copying.
/// Repeated keys remain separate occurrences. This plan neither infers geometry
/// nor visits a checkpoint source; it is not a payload or read-metadata grant.
pub struct EncodedRecipeKeysPlan<'a> {
    recipe: &'a DerivedWeightRecipe,
    count: usize,
    backing_bytes: usize,
    contiguous: bool,
}
impl<'a> EncodedRecipeKeysPlan<'a> {
    /// Returns `None` for an operation outside the encoded compiler's structural
    /// subset. Geometry-dependent casts and permutations are checked later by
    /// that compiler, preserving its validation and error ordering.
    pub fn new(recipe: &'a DerivedWeightRecipe) -> Result<Option<Self>, RecipeError> {
        let overflow = || RecipeError::ArithmeticOverflow("encoded recipe source keys");
        let mut count = 0usize;
        let mut backing_bytes = 0usize;
        let Some(contiguous) = visit(recipe, &mut |key| {
            count = count.checked_add(1).ok_or_else(overflow)?;
            backing_bytes = backing_bytes.checked_add(key.len()).ok_or_else(overflow)?;
            Ok::<_, RecipeError>(())
        })?
        else {
            return Ok(None);
        };
        backing_bytes = backing_bytes
            .checked_add(
                Layout::array::<String>(count)
                    .map_err(|_| overflow())?
                    .size(),
            )
            .ok_or_else(overflow)?;
        Ok(Some(Self {
            recipe,
            count,
            backing_bytes,
            contiguous,
        }))
    }

    pub(super) fn contiguous(&self) -> bool {
        self.contiguous
    }

    /// Requested key/vector backing and fixed constructor/result controls.
    /// Existing recipe storage, thread stack, allocator bookkeeping and custody
    /// construction are separate. The caller compares this before construction.
    pub fn required_bytes<C>(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<PreparedEncodedRecipeKeys<C>>(),
            size_of::<EncodedRecipeKeysBuildError<C>>(),
            size_of::<Result<PreparedEncodedRecipeKeys<C>, EncodedRecipeKeysBuildError<C>>>(),
            size_of::<String>(),
            size_of::<Result<(), TryReserveError>>(),
        ]
        .into_iter()
        .try_fold(self.backing_bytes, usize::checked_add)
    }

    /// Creates exact requested destinations while retaining the supplied custody
    /// after every completed key, including a failed allocation prefix.
    pub fn construct<C>(
        self,
        custody: C,
    ) -> Result<PreparedEncodedRecipeKeys<C>, EncodedRecipeKeysBuildError<C>> {
        let mut owner = PreparedEncodedRecipeKeys {
            keys: Vec::new(),
            _custody: custody,
        };
        let result = (|| {
            owner.keys.try_reserve_exact(self.count)?;
            visit(self.recipe, &mut |key| {
                let mut owned = String::new();
                owned.try_reserve_exact(key.len())?;
                owned.push_str(key);
                owner.keys.push(owned);
                Ok::<_, TryReserveError>(())
            })?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(owner),
            Err(cause) => Err(EncodedRecipeKeysBuildError {
                cause,
                partial: owner,
            }),
        }
    }
}

/// Move-only source occurrences. Borrowers cannot detach keys from their custody.
#[derive(Debug)]
pub struct PreparedEncodedRecipeKeys<C> {
    keys: Vec<String>,
    _custody: C,
}
impl<C> PreparedEncodedRecipeKeys<C> {
    /// Exact left-to-right occurrences, including duplicate and empty names.
    /// Source lookup and recipe inference perform their own semantic validation.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
}

/// A reserve failure retaining its actual completed prefix and custody.
pub struct EncodedRecipeKeysBuildError<C> {
    cause: TryReserveError,
    partial: PreparedEncodedRecipeKeys<C>,
}
impl<C> EncodedRecipeKeysBuildError<C> {
    /// Fully constructed occurrences retained with this failure.
    pub fn completed_keys(&self) -> &[String] {
        &self.partial.keys
    }
}
impl<C> fmt::Debug for EncodedRecipeKeysBuildError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodedRecipeKeysBuildError")
            .field("cause", &self.cause)
            .field("completed_keys", &self.partial.keys.len())
            .finish_non_exhaustive()
    }
}
impl<C> fmt::Display for EncodedRecipeKeysBuildError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<C> std::error::Error for EncodedRecipeKeysBuildError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

// Both sizing and construction use this traversal. The boolean selects the
// existing contiguous fast path; false requires the range-projection compiler.
fn visit<E>(
    recipe: &DerivedWeightRecipe,
    key: &mut impl FnMut(&str) -> Result<(), E>,
) -> Result<Option<bool>, E> {
    use DerivedWeightRecipe as R;
    Ok(Some(match recipe {
        R::Source {
            key: name,
            selection,
        } => {
            key(name)?;
            matches!(selection, TensorSelection::Full)
        }
        R::Concatenate { axis, inputs } | R::Stack { axis, inputs } => {
            let mut contiguous = *axis == 0;
            for input in inputs {
                let Some(child) = visit(input, key)? else {
                    return Ok(None);
                };
                contiguous &= child;
            }
            contiguous
        }
        R::Select { input, .. } => {
            if visit(input, key)?.is_none() {
                return Ok(None);
            }
            false
        }
        R::Reshape { input, .. }
        | R::View { input, .. }
        | R::Transpose { input, .. }
        | R::Cast { input, .. } => {
            let Some(child) = visit(input, key)? else {
                return Ok(None);
            };
            child
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests;
