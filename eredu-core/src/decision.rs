//! Portable inputs and results for non-autoregressive decisions.

/// An ordered single-choice question about the input state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceQuestion {
    /// Instructions supplied to the decision model.
    pub instructions: String,
    /// Options in the order in which they appear in the prompt and result.
    pub options: Vec<ChoiceOption>,
}

/// One choice option with a stable label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceOption {
    /// Label returned alongside the option probability.
    pub label: String,
    /// Optional explanation of when to select the option.
    pub description: Option<String>,
}

/// Calibrated probabilities in the supplied option order.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ChoiceAnswer {
    /// Labels and calibrated probabilities for the answer slot.
    pub options: Vec<ChoiceProbability>,
}

/// One option's calibrated probability.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ChoiceProbability {
    /// Original option label.
    pub label: String,
    /// Probability in the inclusive interval from zero to one.
    pub probability: f64,
}

/// Decision results for one input state and its ordered questions.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionEvaluation<A = DecisionAnswer> {
    /// Answers in caller question order.
    pub answers: Vec<A>,
    /// Input tokens processed across the packed rows, including repeated state.
    pub input_tokens: usize,
    /// Number of generated tokens. Decision inference reports zero.
    pub generated_tokens: usize,
    /// SHA-256 of the exact calibration companion.
    pub calibration_sha256: [u8; 32],
}

/// Results from the single-choice convenience interface.
pub type ChoiceEvaluation = DecisionEvaluation<ChoiceAnswer>;

/// A text question with an explicit decision readout.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionQuestion {
    /// Locate one instance in the supplied image.
    Box {
        /// Description of the object to locate.
        instructions: String,
    },
    /// Locate multiple instances in the supplied image.
    Boxes {
        /// Description of the objects to locate.
        instructions: String,
        /// Maximum number of instances to read, or the checkpoint default.
        max: Option<usize>,
    },
    /// Estimate a numeric value on a bounded range.
    Estimate {
        /// Instructions describing the quantity to estimate.
        instructions: String,
        /// Inclusive lower and upper bounds.
        range: [f64; 2],
        /// Desired spacing of grid values. Omit for the checkpoint default.
        step: Option<f64>,
        /// Optional unit for the estimate.
        unit: Option<String>,
    },
    /// Extract a written numeric value.
    ReadNumber {
        /// Instructions describing the value to find.
        instructions: String,
        /// Unit requested by the caller, if any.
        unit: Option<String>,
    },
    /// Extract a written calendar date or date-time.
    ReadDate {
        /// Instructions describing the date to find.
        instructions: String,
    },
    /// Extract a span from the supplied text state.
    ExtractText {
        /// Instructions describing the span to find.
        instructions: String,
    },
    /// One categorical distribution over the supplied options.
    SingleChoice(ChoiceQuestion),
    /// Independent yes/no decisions for the supplied options.
    MultipleChoice(ChoiceQuestion),
    /// A binary (noul) question.
    YesNo {
        /// Instructions supplied to the decision model.
        instructions: String,
        /// Optional meanings of yes and no, in that order.
        criteria: Option<[String; 2]>,
    },
    /// A question over ordered, zero-based levels.
    Score {
        /// Instructions supplied to the decision model.
        instructions: String,
        /// Descriptions in increasing level order.
        levels: Vec<String>,
    },
}

/// A typed answer to a decision question.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum DecisionAnswer {
    /// A calibrated single-object result.
    Box(BoxDetection),
    /// Calibrated detections and the distribution over their count.
    Boxes {
        /// Present instances in reading order.
        detections: Vec<BoxDetection>,
        /// Calibrated probability of at least one, two, and subsequent instances.
        presence: Vec<f64>,
        /// Whether the last available slot was also present.
        at_least: bool,
    },
    /// A calibrated numeric estimate on the requested grid.
    Estimate {
        /// Calibrated median.
        median: f64,
        /// Calibrated lower and upper endpoints of the central 80% interval.
        interval: [f64; 2],
        /// Unit supplied with the question.
        unit: Option<String>,
    },
    /// A numeric read and its original text.
    Number {
        /// Parsed finite magnitude, when resolved.
        value: Option<f64>,
        /// Supplied unit or explicitly written percent or currency marker.
        unit: Option<String>,
        /// Calibrated source extraction.
        extraction: SpanExtraction,
    },
    /// A calendar read and its original text.
    Date {
        /// Resolved date or date-time in ISO 8601 form.
        value: Option<String>,
        /// Calibrated source extraction.
        extraction: SpanExtraction,
    },
    /// A calibrated text extraction.
    Text(SpanExtraction),
    /// A categorical distribution in option order.
    SingleChoice(ChoiceAnswer),
    /// Independent selection probabilities in option order.
    MultipleChoice {
        /// Options selected at the publisher's probability threshold.
        selected: Vec<String>,
        /// Calibrated probability of selecting each option.
        options: Vec<ChoiceProbability>,
    },
    /// A binary decision.
    YesNo {
        /// Calibrated probability of yes.
        probability_yes: f64,
    },
    /// An ordinal decision.
    Score {
        /// Median level of the calibrated distribution.
        level: usize,
        /// Probability of the most likely level.
        confidence: f64,
        /// Level descriptions and probabilities in increasing order.
        levels: Vec<ChoiceProbability>,
    },
}

/// A source span and its calibrated presence probability.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SpanExtraction {
    /// Extracted text and its location when the answer is present.
    pub span: Option<TextSpan>,
    /// Calibrated probability that the state contains an answer.
    pub found: f64,
}

/// Text selected from the original input state.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TextSpan {
    /// Original source text.
    pub text: String,
    /// Half-open UTF-8 byte range in the supplied text state.
    pub bytes: std::ops::Range<usize>,
}

/// A rectangle expressed as fractions of the original image dimensions.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ImageBox {
    /// Upper edge measured downwards from the top.
    pub top: f64,
    /// Lower edge measured downwards from the top.
    pub bottom: f64,
    /// Left edge measured rightwards from the left.
    pub left: f64,
    /// Right edge measured rightwards from the left.
    pub right: f64,
}

/// A location with calibrated presence and overlap confidence.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BoxDetection {
    /// Rectangle when the object is present.
    pub bounds: Option<ImageBox>,
    /// Probability that the instance is present.
    pub found: f64,
    /// Calibrated confidence in the overlap of the returned rectangle.
    pub confidence: f64,
}
