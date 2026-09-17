//! Portable realtime-frame validation before opaque token materialization.

use eredu_core::{RealtimeFrameForcing, RealtimeInputFrame, RealtimeSpeechConfig};

use crate::TokenDomain;

/// Exact schedule and token domains used to validate portable realtime input.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeIngressContract {
    schedule: RealtimeSpeechConfig,
    text: TokenDomain,
    audio: TokenDomain,
}

impl RealtimeIngressContract {
    /// Creates a contract whose padding IDs are inside the admitted domains.
    pub fn new(
        schedule: RealtimeSpeechConfig,
        text: TokenDomain,
        audio: TokenDomain,
    ) -> Result<Self, RealtimeIngressError> {
        validate_token(schedule.text_padding_token(), text, RealtimeTokenKind::Text)?;
        validate_token(
            schedule.audio_padding_token(),
            audio,
            RealtimeTokenKind::Audio,
        )?;
        Ok(Self {
            schedule,
            text,
            audio,
        })
    }

    /// Returns the exact normalized speech schedule.
    pub const fn schedule(&self) -> &RealtimeSpeechConfig {
        &self.schedule
    }

    /// Returns the complete admitted text-token domain.
    pub const fn text_domain(&self) -> TokenDomain {
        self.text
    }

    /// Returns the complete admitted audio-token domain.
    pub const fn audio_domain(&self) -> TokenDomain {
        self.audio
    }

    /// Inspects borrowed host payloads without allocating a forcing mask or tensor.
    pub fn inspect<'a>(
        &'a self,
        frame: &'a RealtimeInputFrame,
    ) -> Result<RealtimeIngressSource<'a>, RealtimeIngressError> {
        let batch = frame.batch();
        if batch == 0 {
            return Err(RealtimeIngressError::EmptyBatch);
        }
        let input_columns = self.schedule.input_audio_codebooks();
        validate_shape(
            RealtimePayloadKind::InputAudio,
            frame.input_audio_tokens().len(),
            batch,
            input_columns,
        )?;
        for &token in frame.input_audio_tokens() {
            validate_token(token, self.audio, RealtimeTokenKind::Audio)?;
        }

        let generated = self.schedule.generated_audio_codebooks();
        let forced_audio = frame.forced_generated_audio_tokens();
        match (forced_audio, frame.forced_generated_audio_codebooks()) {
            (None, None) => {},
            (None, Some(_)) => return Err(RealtimeIngressError::ForcingMaskWithoutPayload),
            (Some(tokens), mask) => {
                validate_shape(
                    RealtimePayloadKind::ForcedAudio,
                    tokens.len(),
                    batch,
                    generated,
                )?;
                if let Some(mask) = mask.filter(|mask| mask.len() != generated) {
                    return Err(RealtimeIngressError::ForcingMaskCount {
                        expected: generated,
                        actual: mask.len(),
                    });
                }
                for row in tokens.chunks_exact(generated) {
                    for (column, token) in row.iter().enumerate() {
                        if mask.is_none_or(|mask| mask[column]) {
                            validate_token(*token, self.audio, RealtimeTokenKind::Audio)?;
                        }
                    }
                }
            }
        };

        let forced_text = frame.forced_text_tokens();
        if let Some(tokens) = forced_text {
            validate_shape(RealtimePayloadKind::ForcedText, tokens.len(), batch, 1)?;
            for &token in tokens {
                validate_token(token, self.text, RealtimeTokenKind::Text)?;
            }
        }
        Ok(RealtimeIngressSource { contract: self, frame })
    }

    /// Validates and owns the forcing mask for compatibility with ordinary callers.
    /// Managed callers inspect first and materialize the borrowed source after admission.
    pub fn validate<'a>(&'a self, frame: &'a RealtimeInputFrame)
        -> Result<ValidatedRealtimeInput<'a>, RealtimeIngressError> {
        let source = self.inspect(frame)?;
        Ok(ValidatedRealtimeInput { contract:self, frame, forcing:source.owned_forcing() })
    }
}

/// Exact validated host input. Borrowing keeps payload and selected contract
/// inseparable through cold inspection and the actual materializer callback.
#[derive(Clone, Copy)]
pub struct RealtimeIngressSource<'a> {
    contract: &'a RealtimeIngressContract,
    frame: &'a RealtimeInputFrame,
}
/// One matrix borrowed from the validated frame; callers cannot forge a source.
#[derive(Clone, Copy)]
pub struct RealtimeInputMatrix<'a> {
    kind: RealtimePayloadKind,
    values: &'a [i32],
    shape: [usize; 2],
}
impl RealtimeInputMatrix<'_> {
    /// Semantic payload selected by the shared ingress contract.
    pub const fn kind(&self)->RealtimePayloadKind {self.kind}
    /// Original borrowed host values, with no intermediate copy.
    pub const fn values(&self)->&[i32] {self.values}
    /// Exact validated row-major geometry.
    pub const fn shape(&self)->[usize;2] {self.shape}
}
impl<'a> RealtimeIngressSource<'a> {
    /// Actual schedule and token-domain owner.
    pub const fn contract(&self)->&'a RealtimeIngressContract {self.contract}
    /// Actual frame retained by this inspection loan.
    pub const fn frame(&self)->&'a RealtimeInputFrame {self.frame}
    /// Source identity, independent of equal token contents or matrix geometry.
    pub fn same_source(&self,other:&Self)->bool {
        std::ptr::eq(self.contract,other.contract) && std::ptr::eq(self.frame,other.frame)
    }
    /// Exact optional matrix in canonical input/audio-force/text-force order.
    pub fn matrix(&self,kind:RealtimePayloadKind)->Option<RealtimeInputMatrix<'a>> {
        let (values,columns)=match kind {
            RealtimePayloadKind::InputAudio=>(self.frame.input_audio_tokens(),self.contract.schedule.input_audio_codebooks()),
            RealtimePayloadKind::ForcedAudio=>(self.frame.forced_generated_audio_tokens()?,self.contract.schedule.generated_audio_codebooks()),
            RealtimePayloadKind::ForcedText=>(self.frame.forced_text_tokens()?,1),
        };
        Some(RealtimeInputMatrix{kind,values,shape:[self.frame.batch(),columns]})
    }
    /// Number of actual source constructors; absent forcing has no constructor.
    pub fn matrices(&self)->usize {
        1+usize::from(self.frame.forced_generated_audio_tokens().is_some())
            +usize::from(self.frame.forced_text_tokens().is_some())
    }
    /// Heap bytes of the shared materialized policy, excluding native tensors.
    /// Both owned vectors are allocated only after prepare_source succeeds.
    pub fn policy_storage_bytes(&self)->Option<usize> {
        self.contract.schedule.delays().len().checked_mul(std::mem::size_of::<usize>())?
            .checked_add(self.contract.schedule.generated_audio_codebooks().checked_mul(std::mem::size_of::<bool>())?)
    }
    fn owned_forcing(&self)->RealtimeFrameForcing {
        let generated=self.contract.schedule.generated_audio_codebooks();
        let audio=match self.frame.forced_generated_audio_tokens() {
            None=>vec![false;generated],
            Some(_)=>self.frame.forced_generated_audio_codebooks().map_or_else(||vec![true;generated],<[bool]>::to_vec),
        };
        RealtimeFrameForcing::new(self.frame.forced_text_tokens().is_some(),audio)
    }
    /// One shared constructor: source admission precedes owned policy and tensors.
    pub fn materialize<M:RealtimeHostTokenMaterializer>(&self,materializer:&mut M)
        ->Result<MaterializedRealtimeInput<M::Tensor>,M::Error> {
        materializer.prepare_source(self)?;
        let input_audio=materializer.materialize_matrix(self.matrix(RealtimePayloadKind::InputAudio)
            .expect("validated mandatory input"))?;
        let forced_audio=self.matrix(RealtimePayloadKind::ForcedAudio)
            .map(|matrix|materializer.materialize_matrix(matrix)).transpose()?;
        let forced_text=self.matrix(RealtimePayloadKind::ForcedText)
            .map(|matrix|materializer.materialize_matrix(matrix)).transpose()?;
        Ok(MaterializedRealtimeInput{schedule:self.contract.schedule.clone(),batch:self.frame.batch(),
            input_audio,forced_audio,forced_text,forcing:self.owned_forcing(),
            retain_diagnostics:self.frame.retains_diagnostics()})
    }
}

/// A portable frame proven valid before any opaque/native allocation.
pub struct ValidatedRealtimeInput<'a> {
    contract: &'a RealtimeIngressContract,
    frame: &'a RealtimeInputFrame,
    forcing: RealtimeFrameForcing,
}

impl ValidatedRealtimeInput<'_> {
    /// Returns the validated portable frame.
    pub const fn frame(&self) -> &RealtimeInputFrame {
        self.frame
    }

    /// Returns the exact schedule forcing mask derived from validated payloads.
    pub const fn forcing(&self) -> &RealtimeFrameForcing {
        &self.forcing
    }

    /// Converts validated host arrays through one family-blind mechanism.
    pub fn materialize<M: RealtimeHostTokenMaterializer>(
        &self,
        materializer: &mut M,
    ) -> Result<MaterializedRealtimeInput<M::Tensor>, M::Error> {
        RealtimeIngressSource {contract:self.contract,frame:self.frame}.materialize(materializer)
    }
}

/// Narrow backend mechanism for copying one validated host token matrix.
pub trait RealtimeHostTokenMaterializer {
    /// Opaque/native token tensor.
    type Tensor;
    /// Materialization failure.
    type Error;

    /// Source-specific admission hook, before any materialized policy or input.
    /// Ordinary mechanisms need no reservation; counted mechanisms authenticate
    /// their exact pre-admitted frame and keep its owner through completion.
    fn prepare_source(&mut self, _source:&RealtimeIngressSource<'_>)->Result<(),Self::Error> {Ok(())}

    /// Constructs one actual matrix from the borrowed validated source.
    fn materialize_matrix(&mut self,matrix:RealtimeInputMatrix<'_>)->Result<Self::Tensor,Self::Error> {
        self.materialize_i32(matrix.values(),matrix.shape())
    }

    /// Copies one already validated row-major i32 matrix.
    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<Self::Tensor, Self::Error>;
}

/// Opaque input tensors paired with the neutral forcing and diagnostic policy.
pub struct MaterializedRealtimeInput<T> {
    schedule: RealtimeSpeechConfig,
    batch: usize,
    input_audio: T,
    forced_audio: Option<T>,
    forced_text: Option<T>,
    forcing: RealtimeFrameForcing,
    retain_diagnostics: bool,
}

impl<T> MaterializedRealtimeInput<T> {
    /// Returns the exact schedule under which host validation completed.
    pub const fn schedule(&self) -> &RealtimeSpeechConfig {
        &self.schedule
    }

    /// Returns the validated positive batch dimension.
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Returns input-side audio tokens in batch-by-input-codebook shape.
    pub const fn input_audio(&self) -> &T {
        &self.input_audio
    }

    /// Returns optional forced generated-audio tokens.
    pub const fn forced_audio(&self) -> Option<&T> {
        self.forced_audio.as_ref()
    }

    /// Returns optional forced text tokens.
    pub const fn forced_text(&self) -> Option<&T> {
        self.forced_text.as_ref()
    }

    /// Returns the neutral forcing mask.
    pub const fn forcing(&self) -> &RealtimeFrameForcing {
        &self.forcing
    }

    /// Returns whether ordered logits diagnostics were requested.
    pub const fn retains_diagnostics(&self) -> bool {
        self.retain_diagnostics
    }

    /// Consumes opaque tensors and neutral policy.
    pub fn into_parts(
        self,
    ) -> (
        RealtimeSpeechConfig,
        usize,
        T,
        Option<T>,
        Option<T>,
        RealtimeFrameForcing,
        bool,
    ) {
        (
            self.schedule,
            self.batch,
            self.input_audio,
            self.forced_audio,
            self.forced_text,
            self.forcing,
            self.retain_diagnostics,
        )
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Portable host payload whose row-major geometry failed validation.
pub enum RealtimePayloadKind {
    /// Live input-side audio matrix.
    InputAudio,
    /// Optional generated-audio forcing matrix.
    ForcedAudio,
    /// Optional text forcing column.
    ForcedText,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Architecture-selected token domain used for a portable value.
pub enum RealtimeTokenKind {
    /// Text token domain.
    Text,
    /// Audio token domain.
    Audio,
}

fn validate_shape(
    payload: RealtimePayloadKind,
    actual: usize,
    rows: usize,
    columns: usize,
) -> Result<(), RealtimeIngressError> {
    let expected = rows
        .checked_mul(columns)
        .ok_or(RealtimeIngressError::ShapeOverflow { rows, columns })?;
    if actual == expected {
        Ok(())
    } else {
        Err(RealtimeIngressError::PayloadShape {
            payload,
            expected,
            actual,
        })
    }
}

fn validate_token(
    token: i32,
    domain: TokenDomain,
    kind: RealtimeTokenKind,
) -> Result<(), RealtimeIngressError> {
    let token = usize::try_from(token).map_err(|_| RealtimeIngressError::TokenDomain {
        kind,
        token,
        cardinality: domain.cardinality(),
    })?;
    if token < domain.cardinality() {
        Ok(())
    } else {
        Err(RealtimeIngressError::TokenDomain {
            kind,
            token: i32::try_from(token).unwrap_or(i32::MAX),
            cardinality: domain.cardinality(),
        })
    }
}

/// Invalid portable realtime input detected before opaque materialization.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeIngressError {
    /// Realtime input must contain at least one batch row.
    #[error("realtime input batch must be positive")]
    EmptyBatch,
    /// A row-major payload shape overflowed.
    #[error("realtime payload shape {rows}x{columns} overflowed")]
    ShapeOverflow {
        /// Batch rows.
        rows: usize,
        /// Payload columns.
        columns: usize,
    },
    /// A row-major payload has the wrong number of values.
    #[error("realtime {payload:?} payload has {actual} values, expected {expected}")]
    PayloadShape {
        /// Affected payload.
        payload: RealtimePayloadKind,
        /// Required value count.
        expected: usize,
        /// Supplied value count.
        actual: usize,
    },
    /// A partial forcing mask was supplied without audio payloads.
    #[error("realtime generated-audio forcing mask has no payload")]
    ForcingMaskWithoutPayload,
    /// A generated-audio forcing mask has the wrong codebook count.
    #[error("realtime forcing mask has {actual} entries, expected {expected}")]
    ForcingMaskCount {
        /// Expected generated-codebook count.
        expected: usize,
        /// Supplied mask count.
        actual: usize,
    },
    /// A selected token is outside its exact zero-based domain.
    #[error("realtime {kind:?} token {token} is outside 0..{cardinality}")]
    TokenDomain {
        /// Text or audio domain.
        kind: RealtimeTokenKind,
        /// Invalid token value.
        token: i32,
        /// Exclusive domain end.
        cardinality: usize,
    },
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use eredu_core::{RealtimeFrameConvention, RealtimeInputFrame};

    use super::*;

    fn contract() -> RealtimeIngressContract {
        RealtimeIngressContract::new(
            RealtimeSpeechConfig::new(
                4,
                2,
                2,
                2,
                16,
                15,
                RealtimeFrameConvention::FeedbackAlignedHistory,
                vec![0, 1, 2, 1, 2],
            )
            .unwrap(),
            TokenDomain::new(17),
            TokenDomain::new(16),
        )
        .unwrap()
    }

    #[derive(Default)]
    struct Recorder(Vec<(Vec<i32>, [usize; 2])>);

    impl RealtimeHostTokenMaterializer for Recorder {
        type Tensor = usize;
        type Error = Infallible;

        fn materialize_i32(
            &mut self,
            values: &[i32],
            shape: [usize; 2],
        ) -> Result<Self::Tensor, Self::Error> {
            self.0.push((values.to_vec(), shape));
            Ok(self.0.len() - 1)
        }
    }

    #[test]
    fn validation_precedes_every_opaque_materialization() {
        let contract = contract();
        let invalid = RealtimeInputFrame::new(2, vec![1, 2, 3, 99]);
        let mut materializer = Recorder::default();
        assert!(matches!(
            contract.validate(&invalid),
            Err(RealtimeIngressError::TokenDomain { .. })
        ));
        assert!(materializer.0.is_empty());

        let valid = RealtimeInputFrame::new(2, vec![1, 2, 3, 4])
            .with_partially_forced_generated_audio(vec![5, 99, 6, 99], vec![true, false])
            .with_forced_text(vec![7, 8])
            .with_diagnostics();
        let validated = contract.validate(&valid).unwrap();
        assert_eq!(validated.forcing().generated_audio(), &[true, false]);
        let input = validated.materialize(&mut materializer).unwrap();
        assert_eq!(materializer.0.len(), 3);
        assert_eq!(materializer.0[0].1, [2, 2]);
        assert_eq!(materializer.0[1].1, [2, 2]);
        assert_eq!(materializer.0[2].1, [2, 1]);
        assert!(input.retains_diagnostics());
    }

    #[test]
    fn borrowed_source_admission_precedes_materialization_and_preserves_identity() {
        struct Refusing;
        impl RealtimeHostTokenMaterializer for Refusing {
            type Tensor=(); type Error=&'static str;
            fn prepare_source(&mut self,source:&RealtimeIngressSource<'_>)->Result<(),Self::Error> {
                assert_eq!(source.matrices(),3);
                assert_eq!(source.matrix(RealtimePayloadKind::ForcedText).unwrap().shape(),[2,1]);
                Err("refused before construction")
            }
            fn materialize_i32(&mut self,_:&[i32],_:[usize;2])->Result<(),Self::Error> {
                panic!("refused source must never materialize")
            }
        }
        let contract=contract();
        let frame=RealtimeInputFrame::new(2,vec![1,2,3,4])
            .with_partially_forced_generated_audio(vec![5,99,6,99],vec![true,false])
            .with_forced_text(vec![7,8]);
        let source=contract.inspect(&frame).unwrap();
        assert!(source.same_source(&contract.inspect(&frame).unwrap()));
        let other_contract=contract.clone();
        assert!(!source.same_source(&other_contract.inspect(&frame).unwrap()));
        assert_eq!(source.materialize(&mut Refusing).err(),Some("refused before construction"));
        let mut recorder=Recorder::default();
        let input=source.materialize(&mut recorder).unwrap();
        assert_eq!(input.forcing().generated_audio(),&[true,false]);
        assert_eq!(recorder.0,vec![(vec![1,2,3,4],[2,2]),(vec![5,99,6,99],[2,2]),(vec![7,8],[2,1])]);
    }

    #[test]
    fn shapes_masks_and_selected_domains_fail_closed() {
        let contract = contract();
        assert_eq!(
            contract.validate(&RealtimeInputFrame::new(0, vec![])).err(),
            Some(RealtimeIngressError::EmptyBatch)
        );
        assert!(matches!(
            contract.validate(&RealtimeInputFrame::new(2, vec![1, 2, 3])),
            Err(RealtimeIngressError::PayloadShape { .. })
        ));
        assert!(matches!(
            contract.validate(
                &RealtimeInputFrame::new(1, vec![1, 2])
                    .with_partially_forced_generated_audio(vec![3, 4], vec![true])
            ),
            Err(RealtimeIngressError::ForcingMaskCount { .. })
        ));
        assert!(matches!(
            contract.validate(&RealtimeInputFrame::new(1, vec![1, 2]).with_forced_text(vec![17])),
            Err(RealtimeIngressError::TokenDomain { .. })
        ));
    }
}
