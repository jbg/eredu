//! Vision unit preparation and traversal through the shared residency policy.

use std::{collections::BTreeMap, marker::PhantomData, path::Path, sync::Arc};

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::SharedCheckpointSource,
};
use eredu_nn::{Error, NeuralBackend, Tensor};
use eredu_runtime::{
    prepare_resource_bindings, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddressMap,
    ExecutionUnitLayout, LayerwiseAcquireError, LayerwisePolicy, LayerwisePolicyForward,
    ParameterResourceRecipes, WeightBinding,
};

use super::{Declaration, VisionAdvance, VisionExecution, VisionInput, VisionModule};

/// Exact weight bindings and execution addresses for the vision tower.
pub struct PreparedVisionExecution {
    declaration: Declaration,
    source: SharedCheckpointSource,
    addresses: ExecutionUnitAddressMap,
    bindings: Vec<Vec<WeightBinding>>,
}

impl PreparedVisionExecution {
    /// Resolves the selected vision dtype and independent unit scopes from retained metadata.
    pub fn new(
        declaration: Declaration,
        source: &SharedCheckpointSource,
        dtype: RecipeDtype,
    ) -> Result<Self, Error> {
        let recipes = declaration
            .prepare_weight_units(source, dtype)?
            .into_iter()
            .enumerate()
            .map(|(ordinal, unit)| {
                (
                    ordinal,
                    ParameterResourceRecipes {
                        outputs: unit.recipes,
                        aliases: Vec::new(),
                    },
                )
            })
            .collect();
        Self::from_resources(declaration, Arc::clone(source), recipes)
    }

    /// Opens a released GGUF media projector and retains its decoded vision bindings.
    pub fn new_gguf(
        declaration: Declaration,
        projector: &Path,
        dtype: RecipeDtype,
        max_cached_readers: usize,
    ) -> Result<Self, Error> {
        let (source, recipes) =
            super::gguf::retain_projector(projector, &declaration, max_cached_readers)?;
        Self::from_canonical_recipes(declaration, source, &recipes, dtype)
    }

    pub(in crate::lfm3) fn from_canonical_recipes(
        declaration: Declaration,
        source: SharedCheckpointSource,
        recipes: &BTreeMap<String, DerivedWeightRecipe>,
        dtype: RecipeDtype,
    ) -> Result<Self, Error> {
        let resources = declaration
            .units()
            .enumerate()
            .map(|(ordinal, unit)| {
                let outputs = declaration
                    .unit_parameters(unit)?
                    .into_iter()
                    .map(|parameter| {
                        let key = parameter.parameter.id.as_str();
                        let input = recipes.get(key).ok_or_else(|| {
                            Error::backend(format!("missing GGUF projector binding for {key}"))
                        })?;
                        Ok((
                            key.to_owned(),
                            DerivedWeightRecipe::Cast {
                                input: Box::new(input.clone()),
                                dtype: dtype.clone(),
                            },
                        ))
                    })
                    .collect::<Result<_, Error>>()?;
                Ok((
                    ordinal,
                    ParameterResourceRecipes {
                        outputs,
                        aliases: Vec::new(),
                    },
                ))
            })
            .collect::<Result<_, Error>>()?;
        Self::from_resources(declaration, source, resources)
    }

    fn from_resources(
        declaration: Declaration,
        source: SharedCheckpointSource,
        recipes: BTreeMap<usize, ParameterResourceRecipes>,
    ) -> Result<Self, Error> {
        let (source, bindings) = prepare_resource_bindings(source, "lfm3/vision", recipes)
            .map_err(Error::backend_source)?;
        let addresses = addresses(&declaration)?;
        Ok(Self {
            declaration,
            source,
            addresses,
            bindings: bindings.into_values().collect(),
        })
    }

    pub(super) fn from_component(
        declaration: Declaration,
        source: SharedCheckpointSource,
        component: &eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
    ) -> Result<Self, Error> {
        let group = component
            .description()
            .graph()
            .group_index(super::component::VISION_GROUP)
            .ok_or_else(|| Error::backend("VL component omitted its vision group"))?;
        let addresses = addresses(&declaration)?;
        let bindings = component
            .addressed_unit_bindings()
            .filter(|(address, _)| address.group() == group)
            .map(|(_, bindings)| bindings.to_vec())
            .collect::<Vec<_>>();
        if bindings.len() != declaration.units().count() {
            return Err(Error::backend("VL component omitted vision unit bindings"));
        }
        Ok(Self {
            declaration,
            source,
            addresses,
            bindings,
        })
    }

    /// Exact invocation order used for residency sizing.
    pub fn invocations(&self) -> &ExecutionUnitAddressMap {
        &self.addresses
    }

    /// Per-unit physical bindings used for residency sizing.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        &self.bindings
    }

    /// Transfers the prepared policy inputs to a native binding visitor.
    pub fn into_parts(
        self,
    ) -> (
        Declaration,
        SharedCheckpointSource,
        ExecutionUnitAddressMap,
        Vec<Vec<WeightBinding>>,
    ) {
        (self.declaration, self.source, self.addresses, self.bindings)
    }
}

fn addresses(declaration: &Declaration) -> Result<ExecutionUnitAddressMap, Error> {
    let id = "lfm3.vision";
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root(id)], id)
        .map_err(Error::backend_source)?;
    let layout = ExecutionUnitLayout::new(&graph, [declaration.units().count()])
        .map_err(Error::backend_source)?;
    Ok(ExecutionUnitAddressMap::identity(&layout))
}

/// Vision encoder and projector over a selected resident or bounded unit policy.
pub struct PolicyVision<B: NeuralBackend, P> {
    declaration: Declaration,
    addresses: ExecutionUnitAddressMap,
    policy: P,
    backend: PhantomData<fn() -> B>,
}

impl<B: NeuralBackend, P> PolicyVision<B, P> {
    /// Binds traversal to the invocation map retained during weight preparation.
    pub fn new(
        declaration: Declaration,
        invocations: ExecutionUnitAddressMap,
        policy: P,
    ) -> Result<Self, Error> {
        if addresses(&declaration)? != invocations {
            return Err(Error::backend(
                "vision invocation map differs from the declaration",
            ));
        }
        Ok(Self {
            declaration,
            addresses: invocations,
            policy,
            backend: PhantomData,
        })
    }

    /// Returns the bound policy for native resource reporting.
    pub fn policy(&self) -> &P {
        &self.policy
    }
}

/// Observable progress through one admitted vision invocation.
pub enum PolicyVisionProgress<T> {
    /// One unit completed and a subsequent unit is ready.
    Pending,
    /// Encoder activation and projected language rows are ready.
    Complete {
        /// Final encoder activation.
        hidden: T,
        /// Projected language-width rows.
        projected: T,
    },
}

/// Controlled vision traversal borrowing one exclusive weight policy transaction.
/// Dropping an unfinished session cancels the transaction through the policy guard.
pub struct PolicyVisionSession<'a, B, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, VisionModule<B>>,
{
    declaration: &'a Declaration,
    addresses: &'a ExecutionUnitAddressMap,
    policy: Option<LayerwisePolicyForward<'a, B, VisionModule<B>, P>>,
    context: &'a <B::Tensor as Tensor>::Context,
    current: Option<VisionExecution<B::Tensor>>,
    ordinal: usize,
}

impl<B, P> PolicyVision<B, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, VisionModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Begins a controlled pass with no vision unit acquired.
    pub fn begin<'a>(
        &'a mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &'a <B::Tensor as Tensor>::Context,
    ) -> Result<PolicyVisionSession<'a, B, P>, Error> {
        let policy = LayerwisePolicyForward::begin(&mut self.policy, input.pixels, context)
            .map_err(Error::backend_source)?;
        Ok(PolicyVisionSession {
            current: Some(VisionExecution::new(&self.declaration, input)),
            ordinal: 0,
            declaration: &self.declaration,
            addresses: &self.addresses,
            policy: Some(policy),
            context,
        })
    }

    /// Drains the controlled driver and returns encoder activations and projected rows.
    pub fn forward(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        let mut session = self.begin(input, context)?;
        loop {
            if let PolicyVisionProgress::Complete { hidden, projected } = session.advance()? {
                return Ok((hidden, projected));
            }
        }
    }
}

impl<B, P> PolicyVisionSession<'_, B, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, VisionModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Returns the next unit's semantic address while the pass is live.
    pub fn next_unit(&self) -> Option<super::VisionUnit> {
        self.current.as_ref().map(VisionExecution::next_unit)
    }

    /// Completes one unit before advancing. An execution error terminates this pass.
    pub fn advance(&mut self) -> Result<PolicyVisionProgress<B::Tensor>, Error> {
        let current = self
            .current
            .take()
            .ok_or_else(|| Error::backend("LFM3 vision pass is terminal"))?;
        let mut policy = self
            .policy
            .take()
            .expect("a live vision pass owns a policy transaction");
        let address = self
            .addresses
            .logical_addresses()
            .nth(self.ordinal)
            .ok_or_else(|| Error::backend("vision invocation map ended before projection"))?;
        let unit = current.next_unit();
        let module = policy
            .acquire(self.ordinal, address, |context| {
                VisionModule::new(self.declaration, unit, context)
            })
            .map_err(|error| match error {
                LayerwiseAcquireError::Architecture(error) => error,
                LayerwiseAcquireError::Policy(error) => Error::backend_source(error),
            })?;
        let previous = current.input_or_activation();
        match current.advance(module, self.context)? {
            VisionAdvance::Continue(next) => {
                let hidden = next.input_or_activation();
                policy
                    .complete(
                        hidden,
                        next.retained_positions().into_iter().flatten(),
                        std::iter::once(previous),
                    )
                    .map_err(Error::backend_source)?;
                self.policy = Some(policy);
                self.current = Some(next);
                self.ordinal += 1;
                Ok(PolicyVisionProgress::Pending)
            }
            VisionAdvance::Projected(projected) => {
                policy
                    .complete(&projected, std::iter::empty(), std::iter::once(previous))
                    .map_err(Error::backend_source)?;
                policy.finish(&projected).map_err(Error::backend_source)?;
                Ok(PolicyVisionProgress::Complete {
                    hidden: previous.clone(),
                    projected,
                })
            }
        }
    }
}
