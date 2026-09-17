//! Complete-source vocabulary reductions; no shard-local probability is global evidence.
use super::*;
use eredu_core::ObservationPosition;

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

pub(super) fn is_vocabulary(transform: &CaptureTransform) -> bool {
    matches!(
        transform,
        CaptureTransform::TokenScores { .. } | CaptureTransform::TopCandidates { .. }
    )
}

fn complete_slice(slice: &ResolvedCaptureSlice, shape: &[u64]) -> bool {
    slice.shape == shape
        && slice.ends == shape
        && slice.starts.iter().all(|start| *start == 0)
        && slice.strides.iter().all(|stride| *stride == 1)
}

pub(super) fn validate_projection(
    transform: &CaptureTransform,
    projection: &CaptureSlicePartition,
) -> Result<(), CaptureError> {
    if !is_vocabulary(transform) {
        return Ok(());
    }
    let shape = projection.global_shape();
    let valid = !shape.contains(&0)
        && projection.local_shape() == shape
        && complete_slice(projection.global_slice(), shape)
        && projection.fragments().len() == 1
        && projection.fragments().iter().all(|fragment| {
            complete_slice(fragment.local(), shape) && complete_slice(fragment.destination(), shape)
        });
    if !valid {
        return Err(CaptureError::Unsupported(
            "vocabulary reduction requires one complete ordered global logits source".into(),
        ));
    }
    Ok(())
}

fn domain_valid(domain: Option<CandidateDomain>, vocabulary: u64) -> bool {
    domain.is_none_or(|domain| {
        domain.vocabulary == vocabulary
            && domain.allowed_tokens <= vocabulary
            && (!domain.constrained || domain.allowed_tokens < vocabulary)
    })
}

fn candidate_valid(
    candidate: &CaptureCandidate,
    domain: Option<CandidateDomain>,
    vocabulary: u64,
) -> bool {
    u64::from(candidate.token_id) < vocabulary
        && candidate.score.is_finite()
        && match domain {
            None => candidate.allowed,
            Some(domain) if domain.allowed_tokens == 0 => !candidate.allowed,
            Some(domain) if domain.allowed_tokens == vocabulary => candidate.allowed,
            Some(_) => true,
        }
}

fn probability_bounds_valid(
    target: f32,
    alternative: f32,
    probability: f64,
    vocabulary: u64,
) -> bool {
    // The strongest alternative and target together identify the global maximum.
    // Bound the shifted mass independently of log_partition: at extreme logits,
    // adding log mass to the maximum can round away even in F64.
    let target = f64::from(target);
    let alternative = f64::from(alternative);
    let delta = target - target.max(alternative);
    let maximum_log_mass = (vocabulary as f64).ln();
    let minimum_log_mass = (-(target - alternative).abs()).exp().ln_1p();
    // Permit F32 reduction error in the mass and F64 rounding of a large delta,
    // without making the bound depend on the unshifted score magnitude.
    let tolerance = 32.0 * f64::EPSILON * delta.abs().max(1.0)
        + 8.0 * f64::from(f32::EPSILON) * maximum_log_mass.max(1.0);
    probability >= delta - maximum_log_mass - tolerance
        && probability <= delta - minimum_log_mass + tolerance
}

/// Structural and arithmetic consistency only; independent numerical comparisons
/// establish that the producer evaluated the actual model distribution.
pub(super) fn validate_payload(
    selection: &CaptureSelection,
    position: ObservationPosition,
    shape: &[u64],
    payload: &CapturePayload,
) -> Result<(), CaptureError> {
    let value=match payload {
        CapturePayload::Candidates(value)=>VocabularyPayload::Candidates(value),
        CapturePayload::TokenScores(value)=>VocabularyPayload::Scores(value),
        _=>return Err(invalid("vocabulary payload kind differs")),
    };
    if !payload_valid(selection,position,shape,value) {
        return Err(invalid("vocabulary payload differs from its admitted reduction or score semantics"));
    }
    Ok(())
}

/// Borrowed views of the existing owned payloads; no allocation or new reducer.
#[derive(Debug)]
pub(crate) enum VocabularyPayload<'a> {
    Candidates(&'a CaptureCandidates),Scores(&'a CaptureTokenScores),
}
/// Allocation-free source checks shared by ordinary and original receipt readers.
pub(crate) fn payload_valid(selection:&CaptureSelection,position:ObservationPosition,
    shape:&[u64],payload:VocabularyPayload<'_>)->bool {
    let Some(&vocabulary)=shape.last() else{return false};
    if vocabulary==0 {return false;}
    let expected_source = if position == ObservationPosition::AfterIntervention {
        CandidateLogitsSource::Effective
    } else {
        CandidateLogitsSource::Original
    };
    let valid = match (&selection.transform, payload) {
        (CaptureTransform::TopCandidates { count }, VocabularyPayload::Candidates(value)) => {
            value.source == expected_source
                && *count <= vocabulary
                && value.candidates.len() as u64 == *count
                && domain_valid(value.domain, vocabulary)
                && value
                    .candidates
                    .iter()
                    .enumerate()
                    .all(|(index, candidate)| {
                        candidate_valid(candidate, value.domain, vocabulary)
                            && value.candidates[..index]
                                .iter()
                                .all(|previous| previous.token_id != candidate.token_id)
                            && (index == 0 || value.candidates[index - 1].score >= candidate.score)
                    })
        }
        (CaptureTransform::TokenScores { token_ids }, VocabularyPayload::Scores(value)) => {
            value.source == expected_source
                && value.vocabulary == vocabulary
                && value.log_partition.is_finite()
                && value.scores.len() == token_ids.len()
                && domain_valid(value.domain, vocabulary)
                && value.scores.iter().zip(token_ids).all(|(score, id)| {
                    let target = &score.target;
                    // At very large F32 magnitudes, adding the small log mass can
                    // round in F64. Keep the independently stable log probability.
                    let tolerance = 32.0
                        * f64::EPSILON
                        * value
                            .log_partition
                            .abs()
                            .max(f64::from(target.score).abs())
                            .max(1.0);
                    candidate_valid(target, value.domain, vocabulary)
                        && target.token_id == *id
                        && score.log_probability.is_finite()
                        && score.log_probability <= 0.0
                        && score.rank > 0
                        && score.rank <= vocabulary
                        && ((f64::from(target.score) - value.log_partition) - score.log_probability)
                            .abs()
                            <= tolerance
                        && match &score.strongest_alternative {
                            None => {
                                vocabulary == 1 && score.rank == 1 && score.log_probability == 0.0
                            }
                            Some(other) => {
                                vocabulary > 1
                                    && other.token_id != *id
                                    && candidate_valid(other, value.domain, vocabulary)
                                    && ((other.score > target.score) == (score.rank > 1))
                                    && value.log_partition + tolerance >= f64::from(other.score)
                                    && probability_bounds_valid(
                                        target.score,
                                        other.score,
                                        score.log_probability,
                                        vocabulary,
                                    )
                            }
                        }
                })
        }
        _ => false,
    };
    valid
}

/// Typed terminal source derived from the same immutable capture selection.
/// Physical row narrowing is supplied only by a retained prefill schedule.
#[derive(Debug)]
pub(crate) enum CompleteVocabularyGeometry<'a> {
    Candidates(CaptureCandidateGeometry<'a>),
    Scores(CaptureTokenScoreGeometry<'a>),
}
impl<'a> CompleteVocabularyGeometry<'a> {
    pub(crate) fn control_bytes()->Option<usize> {
        use std::mem::{size_of,size_of_val};
        let parts=[size_of::<Self>()*2,size_of::<Result<Option<Self>,CaptureTensorGeometryError>>(),
            size_of::<CaptureCandidateGeometry<'_>>(),size_of::<CaptureTokenScoreGeometry<'_>>(),
            size_of::<Result<CaptureCandidateGeometry<'_>,CaptureTensorGeometryError>>(),
            size_of::<Result<CaptureTokenScoreGeometry<'_>,CaptureTensorGeometryError>>(),
            size_of::<(&AdmittedCapturePlan,usize,CapturePhase,u64,Option<usize>)>(),
            size_of::<CaptureInvocationShape>()*2,size_of::<Option<CaptureInvocationShape>>(),
            size_of::<([usize;3],usize,&CaptureSelection,&eredu_core::ObservationPoint)>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }

    pub(crate) fn prepare(source:&'a AdmittedCapturePlan,index:usize,phase:CapturePhase,prediction:u64,rows:Option<usize>)
        ->Result<Option<Self>,CaptureTensorGeometryError> {
        let selection=source.plan().selections.get(index)
            .ok_or(CaptureTensorGeometryError::SelectionMissing {index})?;
        Ok(match selection.transform {
            CaptureTransform::TopCandidates { .. }=>{
                let value=CaptureCandidateGeometry::prepare(source,index,phase,prediction,None)?;
                Some(Self::Candidates(match rows {Some(rows)=>value.terminal_readout(rows)?,None=>value}))
            }
            CaptureTransform::TokenScores { .. }=>{
                let value=CaptureTokenScoreGeometry::prepare(source,index,phase,prediction,None)?;
                Some(Self::Scores(match rows {Some(rows)=>value.terminal_readout(rows)?,None=>value}))
            }
            _=>{if rows.is_some(){return Err(CaptureTensorGeometryError::Unsupported);}None}
        })
    }
    pub(crate) fn shape(&self)->&[usize;3] {match self {
        Self::Candidates(value)=>value.source_shape(),Self::Scores(value)=>value.source_shape(),
    }}
    pub(crate) fn matches(&self,shape:&[u64])->bool {
        shape.len()==3&&self.shape().iter().zip(shape).all(|(a,b)|u64::try_from(*a).ok()==Some(*b))
    }
}

pub(crate) fn payload_validation_control_bytes()->Option<usize> {
    use std::mem::{size_of,size_of_val};
    let parts=[size_of::<VocabularyPayload<'_>>(),size_of::<(&CaptureSelection,ObservationPosition,&[u64])>(),
        size_of::<Option<CandidateDomain>>(),size_of::<CandidateLogitsSource>(),
        size_of::<(&CaptureCandidate,Option<CandidateDomain>,u64)>(),
        size_of::<(f32,f32,f64,u64)>(),size_of::<[f64;7]>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_,CaptureCandidate>>>(),
        size_of::<std::slice::Iter<'_,CaptureCandidate>>(),
        size_of::<std::iter::Zip<std::slice::Iter<'_,CaptureTokenScore>,std::slice::Iter<'_,u32>>>(),
        size_of::<(usize,&CaptureCandidate,&CaptureTokenScore,&u32,u64,bool)>(),
    ];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
