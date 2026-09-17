//! Exhaustive source-field copy with the existing closed String/Vec worker.
use super::*;
type Error = CapturePlanCopyError;
pub(super) fn plan(
    w: &mut Worker,
    value: &AdmittedInterventionPlan,
) -> Result<AdmittedInterventionPlan, Error> {
    let AdmittedInterventionPlan {
        plan,
        points,
        request,
        invocation_bounds,
        text_origin,
        identity,
        intent_identity,
        artifact_identity,
        session_id,
    } = value;
    let InterventionPlan {
        schema_version,
        operations,
    } = plan;
    Ok(AdmittedInterventionPlan {
        plan: InterventionPlan {
            schema_version: *schema_version,
            operations: w.vector(operations, operation)?,
        },
        points: w.vector(points, point)?,
        request: *request,
        invocation_bounds: *invocation_bounds,
        text_origin: *text_origin,
        identity: w.text(identity)?,
        intent_identity: w.text(intent_identity)?,
        artifact_identity: w.text(artifact_identity)?,
        session_id: w.text(session_id)?,
    })
}
fn operation(
    w: &mut Worker,
    value: &InterventionOperation,
) -> Result<InterventionOperation, Error> {
    let InterventionOperation {
        id,
        target,
        schedule,
        slices,
        action: edit,
        evidence,
    } = value;
    Ok(InterventionOperation {
        id: w.text(id)?,
        target: w.text(target)?,
        schedule: schedule.clone(),
        slices: w.vector(slices, |w, s| {
            let CaptureSlice {
                axis,
                start,
                end,
                stride,
            } = s;
            Ok(CaptureSlice {
                axis: w.text(axis)?,
                start: *start,
                end: *end,
                stride: *stride,
            })
        })?,
        action: action(w, edit)?,
        evidence: match evidence {
            InterventionEvidence::None => InterventionEvidence::None,
            InterventionEvidence::Preview { max_elements } => InterventionEvidence::Preview {
                max_elements: *max_elements,
            },
            InterventionEvidence::Summary => InterventionEvidence::Summary,
        },
    })
}
fn tensor(w: &mut Worker, value: &InterventionTensor) -> Result<InterventionTensor, Error> {
    let InterventionTensor { shape, values } = value;
    Ok(InterventionTensor {
        shape: w.vector(shape, |_, n| Ok(*n))?,
        values: match values {
            InterventionValues::Float32(v) => {
                InterventionValues::Float32(w.vector(v, |_, n| Ok(*n))?)
            }
            InterventionValues::Float16(v) => {
                InterventionValues::Float16(w.vector(v, |_, n| Ok(*n))?)
            }
            InterventionValues::Bfloat16(v) => {
                InterventionValues::Bfloat16(w.vector(v, |_, n| Ok(*n))?)
            }
        },
    })
}
fn action(w: &mut Worker, value: &InterventionAction) -> Result<InterventionAction, Error> {
    use InterventionAction::*;
    Ok(match value {
        Zero { dtype } => Zero { dtype: *dtype },
        Scale { dtype, factor } => Scale {
            dtype: *dtype,
            factor: *factor,
        },
        Mask { dtype, shape, keep } => Mask {
            dtype: *dtype,
            shape: w.vector(shape, |_, n| Ok(*n))?,
            keep: w.vector(keep, |_, n| Ok(*n))?,
        },
        MaskComponents {
            dtype,
            indices,
            keep_selected,
        } => MaskComponents {
            dtype: *dtype,
            indices: w.vector(indices, |_, n| Ok(*n))?,
            keep_selected: *keep_selected,
        },
        Replace { tensor: t } => Replace {
            tensor: tensor(w, t)?,
        },
        Add { tensor: t } => Add {
            tensor: tensor(w, t)?,
        },
        MaskLogits { dtype, token_ids } => MaskLogits {
            dtype: *dtype,
            token_ids: w.vector(token_ids, |_, n| Ok(*n))?,
        },
        ExcludeExperts { expert_ids } => ExcludeExperts {
            expert_ids: w.vector(expert_ids, |_, n| Ok(*n))?,
        },
        ZeroExpertContribution { expert_ids } => ZeroExpertContribution {
            expert_ids: w.vector(expert_ids, |_, n| Ok(*n))?,
        },
        BiasRoutingScores {
            stage,
            expert_ids,
            biases,
        } => BiasRoutingScores {
            stage: *stage,
            expert_ids: w.vector(expert_ids, |_, n| Ok(*n))?,
            biases: w.vector(biases, |_, n| Ok(*n))?,
        },
        ForceExperts { shape, expert_ids } => ForceExperts {
            shape: *shape,
            expert_ids: w.vector(expert_ids, |_, n| Ok(*n))?,
        },
    })
}
fn support(
    w: &mut Worker,
    value: &ObservationSupportStatus,
) -> Result<ObservationSupportStatus, Error> {
    Ok(match value {
        ObservationSupportStatus::Supported => ObservationSupportStatus::Supported,
        ObservationSupportStatus::Conditional(reason) => {
            ObservationSupportStatus::Conditional(w.text(reason)?)
        }
        ObservationSupportStatus::Unsupported(reason) => {
            ObservationSupportStatus::Unsupported(w.text(reason)?)
        }
        ObservationSupportStatus::Unverified(reason) => {
            ObservationSupportStatus::Unverified(w.text(reason)?)
        }
    })
}
fn point(w: &mut Worker, value: &InterventionPoint) -> Result<InterventionPoint, Error> {
    let InterventionPoint {
        path,
        node_id,
        stage,
        axes,
        dtypes,
        operations,
        score_stages,
        prefill,
        decode,
        conditions,
        routing,
        routed_units,
    } = value;
    Ok(InterventionPoint {
        path: w.text(path)?,
        node_id: w.text(node_id)?,
        stage: *stage,
        axes: w.vector(axes, |w, a| {
            let TensorAxis { name, dimension } = a;
            Ok(TensorAxis {
                name: w.text(name)?,
                dimension: dimension.clone(),
            })
        })?,
        dtypes: w.vector(dtypes, |_, n| Ok(*n))?,
        operations: w.vector(operations, |_, n| Ok(*n))?,
        score_stages: w.vector(score_stages, |_, n| Ok(*n))?,
        prefill: support(w, prefill)?,
        decode: support(w, decode)?,
        conditions: w.vector(conditions, |w, s| w.text(s))?,
        routing: routing.clone(),
        routed_units: routed_units
            .as_ref()
            .map(|v| {
                let RoutedUnitInterventionPoint { routing, geometry } = v;
                Ok::<_, Error>(RoutedUnitInterventionPoint {
                    routing: w.text(routing)?,
                    geometry: *geometry,
                })
            })
            .transpose()?,
    })
}

pub(super) fn evidence(
    w: &mut Worker,
    source: &AdmittedInterventionPlan,
) -> Result<Vec<Option<InterventionEvidenceCompanion>>, Error> {
    let mut index = 0usize;
    w.vector(source.plan().operations.as_slice(), |w, operation| {
        let current = index;
        index += 1;
        let point = source.points().get(current).ok_or(Error::Capacity)?;
        crate::capture::InterventionEvidenceCompanion::construct(
            w, source, current, operation, point,
        )
    })
}
