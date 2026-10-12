//! Released encoder image modules using shared portable vision equations.
use super::FamilyConfig;
use crate::gemma4::{ModalityProjector, VisionInput, VisionLayer, VisionState, VisionStatic};
use eredu_nn::{Error, NeuralBackend, Parameterized, Tensor};

/// Pinned patch/position/standardization and image-to-text parameters.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ImageStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// The released image tower under the encoder's physical parameter root.
    pub tower: VisionStatic<B>,
    /// Weightless normalization and learned projection into text width.
    pub projector: ModalityProjector<B>,
}
impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> ImageStatic<B> {
    /// Declares the family's exact image parameters without loading tensors.
    pub fn new(
        config: &FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let vision = config
            .vision
            .as_ref()
            .ok_or_else(|| Error::backend("the admitted diffusion artifact has no image tower"))?;
        let key = "model.encoder.embed_vision.embedding_projection.weight";
        Ok(Self {
            tower: VisionStatic::new_at(vision.clone(), "model.encoder.vision_tower", context)?,
            projector: ModalityProjector::new_at_with_format(
                &config.text,
                "model.encoder.embed_vision",
                vision.hidden_size,
                vision.rms_norm_eps,
                config.text.linear_format_for(key),
                context,
            )?,
        })
    }
    /// Declares one independently materializable released image layer.
    pub fn build_layer(
        config: &FamilyConfig,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionLayer<B>, Error> {
        let vision = config
            .vision
            .as_ref()
            .ok_or_else(|| Error::backend("the admitted diffusion artifact has no image tower"))?;
        if index >= vision.num_hidden_layers as usize {
            return Err(Error::backend(
                "diffusion image layer lies outside the admitted tower",
            ));
        }
        VisionLayer::new_at(vision, index, "model.encoder.vision_tower", context)
    }
    /// Embeds patch data and prepares invocation-local positions and visibility.
    pub fn begin(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        self.tower.begin(input, context)
    }
    /// Applies spatial pooling, learned standardization and image-to-text projection.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.projector
            .forward(&self.tower.finish(hidden, state, context)?, context)
    }
}
