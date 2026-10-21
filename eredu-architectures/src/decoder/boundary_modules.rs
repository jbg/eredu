//! Selected vocabulary and architecture-owned readout resources.
use super::*;

/// Pinned decoder resources with an optional architecture-owned residual boundary.
/// Input-only owners retain vocabulary parameters; output owners retain the
/// readout and the actual tied or untied vocabulary owner.
#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct BoundaryModules<B: NeuralBackend, D: DecoderBoundary<B>> {
    /// Input or tied-output vocabulary storage, when selected.
    pub embeddings: Option<B::Embedding>,
    /// Architecture-owned readout, when selected.
    pub boundary: Option<D>,
    /// Untied vocabulary projection, when selected.
    pub lm_head: Option<B::Linear>,
}
impl<B: NeuralBackend, D: DecoderBoundary<B> + Clone> Clone for BoundaryModules<B, D> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            boundary: self.boundary.clone(),
            lm_head: self.lm_head.clone(),
        }
    }
}
impl<B: NeuralBackend, D: DecoderBoundary<B>> From<StaticModules<B, D>> for BoundaryModules<B, D> {
    fn from(modules: StaticModules<B, D>) -> Self {
        Self {
            embeddings: Some(modules.embeddings),
            boundary: Some(modules.boundary),
            lm_head: modules.lm_head,
        }
    }
}
impl<B: NeuralBackend, D: DecoderBoundary<B>> BoundaryModules<B, D> {
    /// Consumes the aggregate and retains its declared canonical storage roles.
    pub fn select(mut self, bindings: &[OwnedParameterGroupSpec]) -> Result<Self, Error> {
        let owns = |role| {
            bindings
                .iter()
                .any(|group| group.owner().static_storage_role() == Some(role))
        };
        for (role, present) in [
            ("embedding", self.embeddings.is_some()),
            ("norm", self.boundary.is_some()),
            ("output", self.lm_head.is_some()),
        ] {
            if owns(role) && !present {
                return Err(Error::backend(format!(
                    "decoder selection requires unretained pinned role {role:?}"
                )));
            }
        }
        if !owns("embedding") {
            self.embeddings = None;
        }
        if !owns("norm") {
            self.boundary = None;
        }
        if !owns("output") {
            self.lm_head = None;
        }
        Ok(self)
    }

    /// Borrows the selected base-width lookup without residual expansion.
    pub fn embedding_mut(&mut self) -> Result<&mut B::Embedding, Error> {
        self.embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("decoder does not own input embedding"))
    }

    /// Visits exactly the physically retained canonical storage roles.
    pub fn visit<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(embedding) = &self.embeddings {
            visitor.visit("embedding", embedding)?;
        }
        if let Some(boundary) = &self.boundary {
            visitor.visit("norm", boundary)?;
        }
        if let Some(head) = &self.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    /// Mutably visits those same physical resource owners.
    pub fn visit_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(embedding) = &mut self.embeddings {
            visitor.visit_mut("embedding", embedding)?;
        }
        if let Some(boundary) = &mut self.boundary {
            visitor.visit_mut("norm", boundary)?;
        }
        if let Some(head) = &mut self.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }

    pub(crate) fn finish_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self
            .boundary
            .as_mut()
            .ok_or_else(|| Error::backend("decoder does not own residual readout"))?
            .collapse(hidden, context, instrumentation)?;
        DecoderOutput::<B> {
            embedding: self.embeddings.as_mut(),
            head: self.lm_head.as_mut(),
        }
        .project(&hidden, context, instrumentation)
    }
}
impl<B: eredu_nn::DistributedNeuralBackend, D: DecoderBoundary<B>> BoundaryModules<B, D> {
    /// Projects prediction-collapsed values through the selected vocabulary owner.
    pub(crate) fn project_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let output = DecoderOutput::<B> {
            embedding: self.embeddings.as_mut(),
            head: self.lm_head.as_mut(),
        };
        match parallel {
            Some(parallel) => output.project_parallel(hidden, parallel, context, instrumentation),
            None => output.project(hidden, context, instrumentation),
        }
    }
}
