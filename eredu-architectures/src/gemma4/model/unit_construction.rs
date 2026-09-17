//! Select the actual group before allocating its concrete module temporaries.
use super::*;
use crate::decoder::ModuleMetadata;

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    #[inline(never)]
    pub(super) fn build_selected_unit(&self, group: usize, index: usize,
        context: &<B::Tensor as Tensor>::Context) -> Result<Unit<B>, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(&Self, usize, usize, &<B::Tensor as Tensor>::Context)>()?;
        self.validate_unit_index(group, index, metadata)?;
        match group {
            0 => self.build_vision_unit(index, context),
            1 => self.build_audio_unit(index, context),
            2 => self.build_text_unit(index, context),
            _ => Err(metadata.error(format_args!("Gemma 4 has three execution groups"))),
        }
    }

    #[inline(never)]
    fn build_vision_unit(&self, index: usize,
        context: &<B::Tensor as Tensor>::Context) -> Result<Unit<B>, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(&Self, usize, VisionLayer<B>, Unit<B>,
            &<B::Tensor as Tensor>::Context)>()?;
        Ok(Unit::Vision(VisionLayer::new(self.args.vision.as_ref()
            .ok_or_else(|| metadata.error(format_args!("Gemma 4 has no vision config")))?,
            index, context)?))
    }

    #[inline(never)]
    fn build_audio_unit(&self, index: usize,
        context: &<B::Tensor as Tensor>::Context) -> Result<Unit<B>, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(&Self, usize, AudioLayer<B>, Unit<B>,
            &<B::Tensor as Tensor>::Context)>()?;
        Ok(Unit::Audio(AudioLayer::new(self.args.audio.as_ref()
            .ok_or_else(|| metadata.error(format_args!("Gemma 4 has no audio config")))?,
            index, context)?))
    }

    #[inline(never)]
    fn build_text_unit(&self, index: usize,
        context: &<B::Tensor as Tensor>::Context) -> Result<Unit<B>, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(&Self, usize, &super::super::ModelArgs,
            Option<eredu_nn::GroupedGatedProductSpec>, DenseBlock<B>, Unit<B>,
            &<B::Tensor as Tensor>::Context)>()?;
        let args = match &self.parallel_geometry {
            Some(geometry) => geometry.text_block(index).ok_or_else(||
                metadata.error(format_args!("missing rank-local Gemma 4 text geometry {index}")))?,
            None => &self.args.text,
        };
        let routed_spec = self.expert_realization.as_ref()
            .and_then(|plan| plan.unit_spec(TEXT_EXECUTION_GROUP, index)).cloned();
        Ok(Unit::Text(DenseBlock::new_at_with_routed_spec(args, index,
            "model.language_model.layers", routed_spec, context)?))
    }
}
