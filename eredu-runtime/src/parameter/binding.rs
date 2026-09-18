//! Shared prepublication traversal, with ordinary maps or caller-owned finite rows.
use super::*;
use eredu_nn::ParameterMetadataView;
use std::{alloc::Layout, mem::size_of};

trait Values<W, E> {
    type Error;
    fn source(error: eredu_nn::ParameterSourceError) -> Self::Error;

    fn ignore_remaining(&self) -> bool {
        false
    }

    fn read(&mut self, id: &ParameterId) -> Result<&W, Self::Error>;
    fn read_complete(&self) -> Result<(), Self::Error>;
    fn mutable(&mut self, id: &ParameterId) -> Result<(), Self::Error>;
    fn mutable_complete(&self) -> Result<(), Self::Error>;
    fn take(&mut self, id: &ParameterId) -> W;
    fn empty(&self) -> bool;
    fn backend(error: E) -> Self::Error;
}
#[derive(Clone, Copy)]
enum Phase {
    Read,
    Mutable,
    Publish,
}
struct Pass<'a, P, W, E, S: Values<W, E>> {
    values: &'a mut S,
    excluded: &'a dyn Fn(&ParameterId) -> bool,
    validate: &'a dyn Fn(&P, &W) -> Result<(), E>,
    bind: &'a dyn Fn(&mut P, W),
    error: Option<S::Error>,
    phase: Phase,
}
impl<P, W, E, S: Values<W, E>> Pass<'_, P, W, E, S> {
    fn read(&mut self, id: &ParameterId, value: &P) {
        if self.error.is_some() || (self.excluded)(id) {
            return;
        }
        let result = self
            .values
            .read(id)
            .and_then(|weight| (self.validate)(value, weight).map_err(S::backend));
        self.error = result.err();
    }
    fn mutate(&mut self, id: &ParameterId, value: &mut P) {
        if self.error.is_some() || self.values.ignore_remaining() || (self.excluded)(id) {
            return;
        }
        match self.phase {
            Phase::Mutable => self.error = self.values.mutable(id).err(),
            Phase::Publish => (self.bind)(value, self.values.take(id)),
            Phase::Read => unreachable!("immutable binding pass"),
        }
    }
}
impl<'v, P: 'v, W, E, S: Values<W, E>> ParameterVisitor<'v, P> for Pass<'_, P, W, E, S> {
    fn visit(&mut self, metadata: ParameterMetadataView<'_>, value: &'v P) {
        self.read(metadata.id(), value);
    }
}
impl<'v, P: 'v, W, E, S: Values<W, E>> ParameterVisitorMut<'v, P> for Pass<'_, P, W, E, S> {
    fn visit_mut(&mut self, metadata: ParameterMetadataView<'_>, value: &'v mut P) {
        self.mutate(metadata.id(), value);
    }
}
fn bind_values<P: 'static, W, E, M: Parameterized<P>, S: Values<W, E>>(
    module: &mut M,
    mut values: S,
    excluded: &dyn Fn(&ParameterId) -> bool,
    validate: &dyn Fn(&P, &W) -> Result<(), E>,
    bind: &dyn Fn(&mut P, W),
) -> Result<(), S::Error> {
    let mut pass = Pass {
        values: &mut values,
        excluded,
        validate,
        bind,
        error: None,
        phase: Phase::Read,
    };
    module.visit_parameters(&mut pass).map_err(S::source)?;
    if let Some(error) = pass.error {
        return Err(error);
    }
    pass.values.read_complete()?;
    pass.phase = Phase::Mutable;
    module.visit_parameters_mut(&mut pass);
    if let Some(error) = pass.error {
        return Err(error);
    }
    pass.values.mutable_complete()?;
    pass.phase = Phase::Publish;
    module.visit_parameters_mut(&mut pass);
    assert!(
        pass.error.is_none() && pass.values.empty(),
        "prepublication mutable traversal validated complete binding consumption"
    );
    Ok(())
}
struct Ordinary<W> {
    weights: BTreeMap<ParameterId, W>,
    read: BTreeMap<ParameterId, ()>,
    mutable: BTreeMap<ParameterId, ()>,
    unexpected: Vec<ParameterId>,
    duplicate: Option<ParameterId>,
}
impl<W, E: std::error::Error + Send + Sync + 'static> Values<W, E> for Ordinary<W> {
    type Error = ParameterOrchestrationError<E>;
    fn source(error: eredu_nn::ParameterSourceError) -> Self::Error { ParameterOrchestrationError::Source(error) }
    fn ignore_remaining(&self) -> bool {
        self.duplicate.is_some()
    }

    fn read(&mut self, id: &ParameterId) -> Result<&W, Self::Error> {
        if self.read.insert(id.clone(), ()).is_some() {
            return Err(ParameterOrchestrationError::DuplicateParameter {
                parameter: id.clone(),
            });
        }
        self.weights
            .get(id)
            .ok_or_else(|| ParameterOrchestrationError::MissingBinding {
                parameter: id.clone(),
            })
    }
    fn read_complete(&self) -> Result<(), Self::Error> {
        let parameters = self
            .weights
            .keys()
            .filter(|id| !self.read.contains_key(*id))
            .cloned()
            .collect::<Vec<_>>();
        if parameters.is_empty() {
            Ok(())
        } else {
            Err(ParameterOrchestrationError::UnexpectedBindings { parameters })
        }
    }
    fn mutable(&mut self, id: &ParameterId) -> Result<(), Self::Error> {
        if self.duplicate.is_none() {
            if self.mutable.insert(id.clone(), ()).is_some() {
                self.duplicate = Some(id.clone());
            } else if !self.read.contains_key(id) {
                self.unexpected.push(id.clone());
            }
        }
        Ok(())
    }
    fn mutable_complete(&self) -> Result<(), Self::Error> {
        if let Some(parameter) = &self.duplicate {
            return Err(ParameterOrchestrationError::DuplicateParameter {
                parameter: parameter.clone(),
            });
        }
        let mut parameters = self.unexpected.clone();
        parameters.extend(
            self.read
                .keys()
                .filter(|id| !self.mutable.contains_key(*id))
                .cloned(),
        );
        if parameters.is_empty() {
            return Ok(());
        }
        parameters.sort();
        parameters.dedup();
        Err(ParameterOrchestrationError::ParameterTraversalMismatch { parameters })
    }
    fn take(&mut self, id: &ParameterId) -> W {
        self.weights
            .remove(id)
            .expect("prepublication mutable traversal validated every binding identity")
    }
    fn empty(&self) -> bool {
        self.weights.is_empty()
    }
    fn backend(error: E) -> Self::Error {
        ParameterOrchestrationError::Backend(error)
    }
}
pub(super) fn ordinary<P: 'static, W, E, M: Parameterized<P>>(
    module: &mut M,
    weights: BTreeMap<ParameterId, W>,
    excluded: impl Fn(&ParameterId) -> bool,
    validate: impl Fn(&P, &W) -> Result<(), E>,
    bind: impl Fn(&mut P, W),
) -> Result<(), ParameterOrchestrationError<E>>
where
    E: std::error::Error + Send + Sync + 'static,
{
    bind_values(
        module,
        Ordinary {
            weights,
            read: BTreeMap::new(),
            mutable: BTreeMap::new(),
            unexpected: Vec::new(),
            duplicate: None,
        },
        &excluded,
        &validate,
        &bind,
    )
}

/// One final binding value and its borrowed source identity. Traversal flags
/// occupy the same caller-owned row; validation allocates no maps or names.
pub struct PreparedParameterBinding<'a, W> {
    name: &'a str,
    value: Option<W>,
    read: bool,
    mutable: bool,
}
impl<'a, W> PreparedParameterBinding<'a, W> {
    /// Creates a row without cloning the source identity or value.
    pub fn new(name: &'a str, value: W) -> Self {
        Self {
            name,
            value: Some(value),
            read: false,
            mutable: false,
        }
    }
}
/// Fixed prepublication failures for a finite binding source.
#[derive(Debug, thiserror::Error)]
pub enum PreparedParameterBindingError<E> {
    /// Canonical source traversal failed before binding publication.
    #[error(transparent)]
    Source(#[from] eredu_nn::ParameterSourceError),

    /// The source repeated an identity.
    #[error("duplicate prepared binding at row {row}")]
    DuplicateBinding {
        /// Source-local row ordinal.
        row: usize,
    },
    /// A module parameter has no supplied binding.
    #[error("prepared parameter binding is missing")]
    MissingBinding,
    /// An immutable or mutable traversal repeated a supplied identity.
    #[error("duplicate prepared parameter at row {row}")]
    DuplicateParameter {
        /// Source-local row ordinal.
        row: usize,
    },
    /// A supplied value was not visited.
    #[error("unexpected prepared binding at row {row}")]
    UnexpectedBinding {
        /// Source-local row ordinal.
        row: usize,
    },
    /// Mutable traversal disagreed with the validated immutable topology.
    #[error("prepared parameter traversal mismatch")]
    TraversalMismatch,
    /// Backend validation failed before any replacement.
    #[error(transparent)]
    Backend(E),
}
struct Prepared<'a, 'name, W>(&'a mut [PreparedParameterBinding<'name, W>]);
impl<W, E> Values<W, E> for Prepared<'_, '_, W> {
    type Error = PreparedParameterBindingError<E>;
    fn source(error: eredu_nn::ParameterSourceError) -> Self::Error { PreparedParameterBindingError::Source(error) }

    fn read(&mut self, id: &ParameterId) -> Result<&W, Self::Error> {
        let (index, row) = self
            .0
            .iter_mut()
            .enumerate()
            .find(|(_, row)| row.name == id.as_str())
            .ok_or(PreparedParameterBindingError::MissingBinding)?;
        if row.read {
            return Err(PreparedParameterBindingError::DuplicateParameter { row: index });
        }
        row.read = true;
        Ok(row.value.as_ref().expect("unpublished prepared binding"))
    }
    fn read_complete(&self) -> Result<(), Self::Error> {
        match self.0.iter().position(|row| !row.read) {
            Some(row) => Err(PreparedParameterBindingError::UnexpectedBinding { row }),
            None => Ok(()),
        }
    }
    fn mutable(&mut self, id: &ParameterId) -> Result<(), Self::Error> {
        let (index, row) = self
            .0
            .iter_mut()
            .enumerate()
            .find(|(_, row)| row.name == id.as_str())
            .ok_or(PreparedParameterBindingError::TraversalMismatch)?;
        if row.mutable {
            return Err(PreparedParameterBindingError::DuplicateParameter { row: index });
        }
        row.mutable = true;
        Ok(())
    }
    fn mutable_complete(&self) -> Result<(), Self::Error> {
        if self.0.iter().all(|row| row.mutable) {
            Ok(())
        } else {
            Err(PreparedParameterBindingError::TraversalMismatch)
        }
    }
    fn take(&mut self, id: &ParameterId) -> W {
        self.0
            .iter_mut()
            .find(|row| row.name == id.as_str())
            .and_then(|row| row.value.take())
            .expect("prepublication mutable traversal validated every binding identity")
    }
    fn empty(&self) -> bool {
        self.0.iter().all(|row| row.value.is_none())
    }
    fn backend(error: E) -> Self::Error {
        PreparedParameterBindingError::Backend(error)
    }
}
/// Binds caller-owned final rows through the same validation/publication worker
/// as ordinary materialization. No binding is changed on validation failure.
///
/// The caller owns row storage, source identities, callback storage and retained
/// values. This worker grants no admission authority or completeness guarantee
/// for arbitrary user-defined traversal callbacks.
pub fn bind_prepared_parameter_values<P: 'static, W, E, M: Parameterized<P>>(
    module: &mut M,
    rows: &mut [PreparedParameterBinding<'_, W>],
    excluded: impl Fn(&ParameterId) -> bool,
    validate: impl Fn(&P, &W) -> Result<(), E>,
    bind: impl Fn(&mut P, W),
) -> Result<(), PreparedParameterBindingError<E>> {
    for index in 0..rows.len() {
        if rows[..index].iter().any(|row| row.name == rows[index].name) {
            return Err(PreparedParameterBindingError::DuplicateBinding { row: index });
        }
        rows[index].read = false;
        rows[index].mutable = false;
        if rows[index].value.is_none() {
            return Err(PreparedParameterBindingError::MissingBinding);
        }
    }
    bind_values(module, Prepared(rows), &excluded, &validate, &bind)
}
/// Exact requested row storage and the shared fixed traversal controls. Caller
/// callback captures, module/source storage, and native values are separate.
pub fn prepared_parameter_binding_control_bytes<P: 'static, W: 'static, E: 'static>(
    rows: usize,
) -> Option<usize> {
    Layout::array::<PreparedParameterBinding<'static, W>>(rows)
        .ok()?
        .size()
        .checked_add(size_of::<Vec<PreparedParameterBinding<'static, W>>>())?
        .checked_add(size_of::<Prepared<'static, 'static, W>>())?
        .checked_add(size_of::<
            Pass<'static, P, W, E, Prepared<'static, 'static, W>>,
        >())?
        .checked_add(size_of::<Result<(), PreparedParameterBindingError<E>>>())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_binding_validation_is_atomic_and_publishes_complete_rows() {
        let mut module = vec![
            eredu_nn::Parameter::new(eredu_nn::ParameterSpec::trainable("first").unwrap(), 1i32),
            eredu_nn::Parameter::new(eredu_nn::ParameterSpec::trainable("second").unwrap(), 2i32),
        ];
        let mut rows = [
            PreparedParameterBinding::new("first", 10),
            PreparedParameterBinding::new("second", 20),
        ];
        let error = bind_prepared_parameter_values(
            &mut module,
            &mut rows,
            |_| false,
            |old, _| {
                if *old == 2 {
                    Err("second refused")
                } else {
                    Ok(())
                }
            },
            |slot, value| *slot = value,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PreparedParameterBindingError::Backend("second refused")
        ));
        assert_eq!((*module[0].as_ref(), *module[1].as_ref()), (1, 2));
        assert!(rows.iter().all(|row| row.value.is_some()));
        bind_prepared_parameter_values(
            &mut module,
            &mut rows,
            |_| false,
            |_, _| Ok::<_, std::convert::Infallible>(()),
            |slot, value| *slot = value,
        )
        .unwrap();
        assert_eq!((*module[0].as_ref(), *module[1].as_ref()), (10, 20));
        assert!(rows.iter().all(|row| row.value.is_none()));
    }
}
