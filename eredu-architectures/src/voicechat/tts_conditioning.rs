//! Character-aware token conditioning and the released EAR-TTS gated fusion.
//!
//! Token strings and saved integer flags are retained preparation data. Mapping
//! never decodes text or reopens tokenizer artifacts during neural execution.

use std::collections::BTreeMap;

use eredu_nn::{
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, Parameter, Parameterized, Tensor,
};

use super::tts_mog::{f32_shape, linear, parameter, TtsError};

/// Admitted checkpoint flags, including the source's two distinct padding IDs.
#[derive(Debug, Clone)]
pub struct CharacterFlags {
    /// Word-start/special=0 and continuation=1, followed by custom-pad zero.
    pub continuation: Vec<i32>,
    /// Source continuation padding index, equal to vocabulary length.
    pub continuation_pad: i32,
    /// Regular=0, BOS=1 and EOS=2; source checkpoint semantics are authoritative.
    pub special: Vec<i32>,
    /// Source special padding index, equal to the highest base vocabulary ID.
    pub special_pad: i32,
}

/// Host character mapping built from exact tokenizer vocabulary and saved flags.
#[derive(Debug, Clone)]
pub struct CharacterMapping {
    characters: Vec<char>,
    token_characters: Vec<Vec<i32>>,
    flags: CharacterFlags,
}

impl CharacterMapping {
    /// Orders single-character tokens by token ID, then filters each token's
    /// characters through that alphabet exactly as the reference `build_vocabs`.
    pub fn new(
        vocabulary: &BTreeMap<u32, String>,
        flags: CharacterFlags,
    ) -> Result<Self, TtsError> {
        let count = i32::try_from(vocabulary.len())
            .map_err(|_| TtsError::Configuration("character vocabulary is too large"))?;
        if count == 0
            || vocabulary.keys().copied().ne(0..count as u32)
            || flags.continuation_pad != count
            || flags.special_pad != count - 1
            || flags.continuation.len() != count as usize + 1
            || flags.special.len() != count as usize
            || flags
                .continuation
                .iter()
                .any(|value| !(0..=1).contains(value))
            || flags.special.iter().any(|value| !(0..=2).contains(value))
            || flags.continuation.last() != Some(&0)
            || flags.special.last() != Some(&0)
        {
            return Err(TtsError::Configuration(
                "invalid character vocabulary or saved token flags",
            ));
        }
        let characters = vocabulary
            .values()
            .filter_map(|token| {
                let mut chars = token.chars();
                let first = chars.next()?;
                chars.next().is_none().then_some(first)
            })
            .collect::<Vec<_>>();
        if characters.is_empty() || characters.len() >= i32::MAX as usize {
            return Err(TtsError::Configuration("invalid character alphabet"));
        }
        let alphabet = characters
            .iter()
            .enumerate()
            .map(|(index, character)| (*character, index as i32))
            .collect::<BTreeMap<_, _>>();
        if alphabet.len() != characters.len() {
            return Err(TtsError::Configuration(
                "duplicate single-character vocabulary entries",
            ));
        }
        let mut token_characters = vocabulary
            .values()
            .map(|token| {
                token
                    .chars()
                    .filter_map(|character| alphabet.get(&character).copied())
                    .collect()
            })
            .collect::<Vec<Vec<i32>>>();
        token_characters.push(vec![characters.len() as i32]);
        Ok(Self {
            characters,
            token_characters,
            flags,
        })
    }

    /// Character alphabet ordered by source token IDs; padding follows its end.
    pub fn characters(&self) -> &[char] {
        &self.characters
    }

    /// Maps selected subwords to a bounded rectangular batch of character IDs.
    /// Unknown positive IDs have empty character sequences and zero flag rows.
    pub fn prepare(
        &self,
        tokens: &[i32],
        mask: Option<&[bool]>,
        batch: i32,
        time: i32,
        max_character_cells: usize,
    ) -> Result<CharacterBatch, TtsError> {
        let cells = batch
            .checked_mul(time)
            .filter(|count| batch > 0 && time > 0 && *count > 0)
            .ok_or(TtsError::Shape("invalid character token geometry"))?
            as usize;
        if tokens.len() != cells
            || mask.is_some_and(|mask| mask.len() != cells)
            || tokens.iter().any(|id| *id < 0)
        {
            return Err(TtsError::Shape(
                "character tokens or mask do not match batch/time",
            ));
        }
        let selected = mask
            .map(<[bool]>::to_vec)
            .unwrap_or_else(|| vec![true; cells]);
        let sequences = tokens
            .iter()
            .zip(&selected)
            .filter(|(_, selected)| **selected)
            .map(|(id, _)| {
                self.token_characters
                    .get(*id as usize)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
            })
            .collect::<Vec<_>>();
        let rows = i32::try_from(sequences.len())
            .map_err(|_| TtsError::Shape("too many character rows"))?;
        let columns = sequences.iter().map(|row| row.len()).max().unwrap_or(0);
        let packed_cells = sequences
            .len()
            .checked_mul(columns)
            .filter(|cells| *cells <= max_character_cells && *cells <= i32::MAX as usize)
            .ok_or(TtsError::Shape("character preparation capacity exceeded"))?;
        let columns_i32 = i32::try_from(columns)
            .map_err(|_| TtsError::Shape("character sequence is too long"))?;
        let mut ids = vec![self.characters.len() as i32; packed_cells];
        let mut lengths = Vec::with_capacity(sequences.len());
        for (index, sequence) in sequences.iter().enumerate() {
            ids[index * columns..index * columns + sequence.len()].copy_from_slice(sequence);
            lengths.push(sequence.len() as i32);
        }
        let continuation = tokens
            .iter()
            .map(|id| self.flags.continuation[(*id).min(self.flags.continuation_pad) as usize])
            .collect();
        let special = tokens
            .iter()
            .map(|id| self.flags.special[(*id).min(self.flags.special_pad) as usize])
            .collect();
        Ok(CharacterBatch {
            batch,
            time,
            rows,
            columns: columns_i32,
            alphabet: self.characters.len() as i32,
            ids,
            lengths,
            selected,
            continuation,
            special,
        })
    }
}

/// Packed valid subwords plus immutable scatter/flag data for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterBatch {
    batch: i32,
    time: i32,
    rows: i32,
    columns: i32,
    alphabet: i32,
    ids: Vec<i32>,
    lengths: Vec<i32>,
    selected: Vec<bool>,
    continuation: Vec<i32>,
    special: Vec<i32>,
}
impl CharacterBatch {
    /// Checks packed input compatibility before acquiring learned parameters.
    pub fn validate_dimensions(&self, dimensions: CharacterDimensions) -> Result<(), TtsError> {
        dimensions.validate()?;
        if self.alphabet != dimensions.characters || self.rows == 0 || self.columns == 0 {
            return Err(TtsError::Shape(
                "character batch has no encodable rows or mismatched alphabet",
            ));
        }
        Ok(())
    }
    /// Packed character IDs, row-major; padding equals character alphabet length.
    pub fn ids(&self) -> &[i32] {
        &self.ids
    }
    /// Number of unpadded characters in each selected subword.
    pub fn lengths(&self) -> &[i32] {
        &self.lengths
    }
    /// Shape `[selected_subwords, maximum_character_length]`.
    pub fn shape(&self) -> [i32; 2] {
        [self.rows, self.columns]
    }
    /// Original token batch and time shape.
    pub fn token_shape(&self) -> [i32; 2] {
        [self.batch, self.time]
    }
    /// True when no subword is selected; the ingress owner skips character encoding.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
    /// Boolean attention mask for the bidirectional character backbone.
    pub fn attention_mask<T: Tensor>(&self, context: &T::Context) -> Result<T, TtsError> {
        let mask = self
            .lengths
            .iter()
            .flat_map(|length| (0..self.columns).map(move |column| i32::from(column < *length)))
            .collect::<Vec<_>>();
        Ok(T::from_i32_slice(&mask, &self.shape(), context)?.equal_i32(1, context)?)
    }
}

/// Character table, projection and optional released flag embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterDimensions {
    /// Alphabet size excluding the padding row.
    pub characters: i32,
    /// Character backbone width.
    pub width: i32,
    /// Projected speech-backbone width.
    pub output_width: i32,
    /// Adds checkpoint continuation embeddings after scatter.
    pub use_continuation: bool,
    /// Adds checkpoint BOS/EOS embeddings after scatter.
    pub use_special: bool,
}

impl CharacterDimensions {
    /// Checks the character/flag geometry without materializing parameters.
    pub fn validate(self) -> Result<(), TtsError> {
        let d = self;
        if d.characters <= 0
            || d.characters.checked_add(1).is_none()
            || d.width <= 0
            || d.output_width <= 0
            || ((d.use_continuation || d.use_special) && d.width != d.output_width)
        {
            return Err(TtsError::Configuration(
                "invalid character encoder dimensions",
            ));
        }
        Ok(())
    }
}

/// Character parameters, independent of the bidirectional blocks and their residency.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CharacterEmbedding<B: NeuralBackend> {
    tokens: Parameter<B::Tensor>,
    projection: B::Linear,
    continuation: Option<Parameter<B::Tensor>>,
    special: Option<Parameter<B::Tensor>>,
    #[parameter(skip)]
    dimensions: CharacterDimensions,
    #[parameter(skip)]
    local_width: i32,
    #[parameter(skip)]
    local_output: i32,
}
impl<B: NeuralBackend> CharacterEmbedding<B> {
    /// Declares canonical character and flag parameters without a second backbone.
    pub fn new(
        dimensions: CharacterDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::new_local(
            dimensions,
            dimensions.width,
            dimensions.output_width,
            context,
        )
    }

    pub(super) fn new_local(
        dimensions: CharacterDimensions,
        local_width: i32,
        local_output: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        dimensions.validate()?;
        if local_width <= 0
            || local_width > dimensions.width
            || local_output <= 0
            || local_output > dimensions.output_width
        {
            return Err(TtsError::Configuration("invalid character partition"));
        }
        let d = dimensions;
        let table = |name: &str, rows: i32| -> Result<_, TtsError> {
            Ok(Parameter::unloaded(
                parameter(name, "voicechat.tts.character.static")?,
                &[rows, d.width],
                context,
            )?)
        };
        Ok(Self {
            tokens: Parameter::unloaded(
                parameter(
                    "embed_subword.embed_tokens.weight",
                    "voicechat.tts.character.static",
                )?,
                &[d.characters + 1, local_width],
                context,
            )?,
            projection: linear::<B>(
                "embed_subword.proj_embedding",
                "voicechat.tts.character.static",
                d.width,
                local_output,
                context,
            )?,
            continuation: d
                .use_continuation
                .then(|| table("embed_subword.subword_flag_emb.cont_emb.weight", 2))
                .transpose()?,
            special: d
                .use_special
                .then(|| table("embed_subword.bos_eos_emb.special_emb.weight", 3))
                .transpose()?,
            dimensions,
            local_width,
            local_output,
        })
    }

    pub(super) fn geometry(&self) -> (CharacterDimensions, i32, i32) {
        (self.dimensions, self.local_width, self.local_output)
    }

    fn validate(&self, batch: &CharacterBatch) -> Result<(), TtsError> {
        batch.validate_dimensions(self.dimensions)?;
        super::precision::require_float32(self, self.tokens.as_ref()).map_err(TtsError::Shape)
    }

    /// Looks up packed character embeddings; backbone scaling belongs to the backbone.
    pub fn embed(
        &self,
        batch: &CharacterBatch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.validate(batch)?;
        let ids = B::Tensor::from_i32_slice(&batch.ids, &[batch.rows * batch.columns], context)?;
        Ok(self
            .tokens
            .as_ref()
            .take_axis(&ids, 0, context)?
            .reshape(&[batch.rows, batch.columns, self.local_width], context)?)
    }

    /// Pools masked hidden states, projects, scatters, then adds saved token flags.
    /// Flag additions include unselected token positions, matching the source.
    pub fn finish(
        &mut self,
        batch: &CharacterBatch,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.finish_with_projection(batch, hidden, context, Ok)
    }

    pub(super) fn finish_with_projection<F>(
        &mut self,
        batch: &CharacterBatch,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        gather: F,
    ) -> Result<B::Tensor, TtsError>
    where
        F: FnOnce(B::Tensor) -> Result<B::Tensor, TtsError>,
    {
        self.validate(batch)?;
        f32_shape(hidden, self.dimensions.width)?;
        if hidden.shape()[..2] != batch.shape() {
            return Err(TtsError::Shape(
                "character backbone output does not match packed inputs",
            ));
        }
        let values = batch
            .lengths
            .iter()
            .flat_map(|length| {
                (0..batch.columns).map(move |column| if column < *length { 1.0 } else { 0.0 })
            })
            .collect::<Vec<_>>();
        let mask = B::Tensor::from_f32_slice(&values, &[batch.rows, batch.columns, 1], context)?;
        let denominator = B::Tensor::from_f32_slice(
            &batch
                .lengths
                .iter()
                .map(|length| (*length).max(1) as f32)
                .collect::<Vec<_>>(),
            &[batch.rows, 1],
            context,
        )?;
        let mean = B::Tensor::sum_axis(&hidden.multiply(&mask, context)?, 1, false, context)?
            .divide(&denominator, context)?;
        let projected = gather(self.projection.forward(&mean, context)?)?;
        let scatter = B::Tensor::from_i32_slice(
            &batch
                .selected
                .iter()
                .map(|selected| i32::from(*selected))
                .collect::<Vec<_>>(),
            &batch.token_shape(),
            context,
        )?
        .equal_i32(1, context)?;
        let mut output = B::Tensor::full_f32(
            0.0,
            &[batch.batch, batch.time, self.dimensions.output_width],
            context,
        )?
        .masked_scatter(&scatter, &projected, context)?;
        for (table, ids) in [
            (&self.continuation, &batch.continuation),
            (&self.special, &batch.special),
        ] {
            if let Some(table) = table {
                let ids = B::Tensor::from_i32_slice(ids, &[batch.batch * batch.time], context)?;
                let flags = table
                    .as_ref()
                    .take_axis(&ids, 0, context)?
                    .reshape(output.shape(), context)?;
                output = output.add(&flags, context)?;
            }
        }
        Ok(output)
    }

    /// Executes conditioning around the caller's ordinary character backbone.
    pub fn forward<F>(
        &mut self,
        batch: &CharacterBatch,
        backbone: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>
    where
        F: FnOnce(&B::Tensor, &B::Tensor) -> Result<B::Tensor, TtsError>,
    {
        let embeddings = self.embed(batch, context)?;
        let mask = batch.attention_mask::<B::Tensor>(context)?;
        let hidden = backbone(&embeddings, &mask)?;
        self.finish(batch, &hidden, context)
    }
}

/// Released Float32 projected audio/text fusion with sigmoid gates and offset RMSNorm.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct GatedFusion<B: NeuralBackend> {
    audio: B::Linear,
    text: B::Linear,
    gate: Parameter<B::Tensor>,
    residual_scale: Parameter<B::Tensor>,
    norm: B::Normalization,
    #[parameter(skip)]
    width: i32,
    #[parameter(skip)]
    local_output: i32,
}
impl<B: NeuralBackend> GatedFusion<B> {
    /// Declares the released 31-codebook divisor and final RMS epsilon `1e-6`.
    pub fn new(width: i32, context: &<B::Tensor as Tensor>::Context) -> Result<Self, TtsError> {
        Self::new_local(width, width, context)
    }

    pub(super) fn new_local(
        width: i32,
        local_output: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        if width <= 0 || local_output <= 0 || local_output > width {
            return Err(TtsError::Configuration("invalid gated fusion width"));
        }
        let prefix = "gated_fusion_audio_text";
        let group = "voicechat.tts.fusion";
        let projection = |name: &str| {
            B::linear(
                LinearSpec {
                    input: width,
                    output: local_output,
                    weight: parameter(&format!("{prefix}.{name}.weight"), group)?,
                    bias: Some(parameter(&format!("{prefix}.{name}.bias"), group)?),
                    format: eredu_nn::LinearFormatSpec::unscaled(
                        eredu_checkpoint::LinearFormat::Dense,
                    )?,
                },
                context,
            )
        };
        Ok(Self {
            audio: projection("audio_proj")?,
            text: projection("text_proj")?,
            gate: Parameter::unloaded(
                parameter(&format!("{prefix}.gate"), group)?,
                &[width],
                context,
            )?,
            residual_scale: Parameter::unloaded(
                parameter(&format!("{prefix}.residual_scale"), group)?,
                &[],
                context,
            )?,
            norm: B::normalization(
                NormalizationConstructionSpec {
                    dimensions: width,
                    epsilon: 1e-6,
                    groups: Some(1),
                    scale: NormalizationScale::LearnedOffset {
                        weight: parameter(&format!("{prefix}.final_norm.weight"), group)?,
                        offset: 1.0,
                    },
                },
                context,
            )?,
            width,
            local_output,
        })
    }

    pub(super) fn geometry(&self) -> (i32, i32) {
        (self.width, self.local_output)
    }

    /// Projects audio divided by 31 and text, mixes featurewise, then normalizes.
    pub fn forward(
        &mut self,
        audio: &B::Tensor,
        text: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        self.forward_with_projection(audio, text, context, Ok)
    }

    pub(super) fn forward_with_projection<F>(
        &mut self,
        audio: &B::Tensor,
        text: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        gather: F,
    ) -> Result<B::Tensor, TtsError>
    where
        F: Fn(B::Tensor) -> Result<B::Tensor, TtsError>,
    {
        f32_shape(audio, self.width)?;
        f32_shape(text, self.width)?;
        if audio.shape() != text.shape() {
            return Err(TtsError::Shape("audio/text fusion shape mismatch"));
        }
        super::precision::require_float32(self, audio).map_err(TtsError::Shape)?;
        let divisor = B::Tensor::full_f32(31.0, &[], context)?;
        let audio = gather(
            self.audio
                .forward(&audio.divide(&divisor, context)?, context)?,
        )?;
        let text = gather(self.text.forward(text, context)?)?;
        let gate = B::sigmoid(self.gate.as_ref().clone(), context)?;
        let scale = B::sigmoid(self.residual_scale.as_ref().clone(), context)?;
        let complement = B::Tensor::full_f32(1.0, &[], context)?.subtract(&gate, context)?;
        let mixed = audio
            .multiply(&gate, context)?
            .add(&text.multiply(&complement, context)?, context)?
            .multiply(&scale, context)?;
        Ok(self.norm.forward(&mixed, context)?)
    }
}
