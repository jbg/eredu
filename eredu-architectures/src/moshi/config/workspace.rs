//! Exact cold copies of the same owned transformer configuration.
use super::*;
use eredu_nn::{Error,workspace::{WorkspaceContext,WorkspaceMetadataError}};
use std::mem::{size_of,size_of_val};

impl MoshiTransformerConfig {
    fn clone_workspace(&self,context:&WorkspaceContext)->Result<Self,Error> {
        let frames=[size_of::<Self>(),size_of::<Result<Self,Error>>(),
            size_of::<LayerSchedule<AttentionPolicy>>(),size_of::<Vec<AttentionPolicy>>(),
            size_of::<(&Self,&WorkspaceContext)>(),size_of::<eredu_core::LayerScheduleError>()];
        context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        let model_identity=context.metadata_string(format_args!("{}",self.model_identity))?;
        let parameter_root=context.metadata_string(format_args!("{}",self.parameter_root))?;
        let mut policies=context.metadata_vec(self.attention_schedule.len())?;
        policies.extend(self.attention_schedule.iter().copied());
        if policies.capacity()!=policies.len() {
            context.charge_metadata(std::alloc::Layout::array::<AttentionPolicy>(policies.len())
                .map_err(|_|WorkspaceMetadataError::Overflow)?.size())?;
        }
        let attention_schedule=LayerSchedule::new(policies.len(),policies)
            .map_err(|cause|context.metadata_source(cause))?;
        Ok(Self {model_identity,parameter_root,attention_schedule,
            hidden_size:self.hidden_size,num_hidden_layers:self.num_hidden_layers,
            num_attention_heads:self.num_attention_heads,head_dim:self.head_dim,
            feed_forward_size:self.feed_forward_size,gated_hidden_size:self.gated_hidden_size,
            context:self.context,attention_window:self.attention_window,rope_base:self.rope_base,
            rms_norm_epsilon:self.rms_norm_epsilon,positional_encoding:self.positional_encoding,
            vocabulary_size:self.vocabulary_size,native_quantization:self.native_quantization,
            parallel_local:self.parallel_local})
    }
    pub(in crate::moshi) fn with_parallel_geometry_workspace(&self,attention_heads:i32,
        gated_hidden_size:i32,context:&WorkspaceContext)->Result<Self,Error> {
        self.validate_parallel_geometry(attention_heads,gated_hidden_size,
            |message|context.metadata_error(message))?;
        let mut local=self.clone_workspace(context)?;
        local.num_attention_heads=attention_heads;
        local.gated_hidden_size=gated_hidden_size;
        local.parallel_local=true;
        local.validate_with_diagnostic(|message|context.metadata_error(message))?;
        Ok(local)
    }
}
impl MoshiConfig {
    pub(in crate::moshi) fn depth_transformer_workspace(&self,codebook:usize,
        context:&WorkspaceContext)->Result<MoshiTransformerConfig,Error> {
        let source=self.depth_template_for(codebook,|message|context.metadata_error(message))?;
        // Keep the ordinary clone-and-replace population: both copied old names
        // and their replacements are paid, with no source or budget inference.
        let mut value=source.clone_workspace(context)?;
        value.parameter_root=context.metadata_string(format_args!("depformer.slices.{codebook}.transformer"))?;
        value.model_identity=context.metadata_string(format_args!("moshi.depth.{codebook}"))?;
        Ok(value)
    }
}
