//! Native component factories used by portable text binding.

use super::*;

impl eredu_core::checkpoint_mechanisms::TextComponentResources for MlxLocalTextBinding<'_> {
    type Values = super::super::retained_session::MlxTextValues;
    type Components = crate::composition::mlx::component_binding::MlxComponentBinder;

    fn text_components(&self) -> Self::Components {
        Self::Components::new(self.weights_stream, self.stream)
    }

    fn component_weights(&self) -> &Stream {
        self.weights_stream
    }

    fn component_execution(&self) -> &Stream {
        self.stream
    }
}
