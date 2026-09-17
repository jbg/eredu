//! Logical operation evidence over the same physical capture/transaction owner.
use super::*;
use eredu_core::{
    intervention::{InterventionOutcome, InterventionRecord, RoutedUnitInterventionReceipt},
    speculative::SpeculativePrefillInterventionReduction as Entry,
};

use crate::working_memory::OriginalInterventionSource;
use eredu_core::intervention::AdmittedInterventionPlan;

#[derive(Clone, Copy)]
enum Source<'a> { Ordinary(&'a AdmittedInterventionPlan), Original(&'a OriginalInterventionSource) }
impl<'a> Source<'a> {
    fn plan(self)->&'a AdmittedInterventionPlan {
        match self {Self::Ordinary(plan)=>plan,Self::Original(source)=>source.plan().admission()}
    }
    fn evidence(self,index:usize)->Result<EvidenceRows<'a>,ConstructionError> {
        let plan=self.plan();
        let operation=&plan.plan().operations[index];let point=&plan.points()[index];
        match self {
            Self::Ordinary(_)=>Ok(EvidenceRows::Owned(crate::intervention::evidence_selections(operation,point))),
            Self::Original(source)=> {
                let companion=source.plan().evidence(index);
                if companion.is_some_and(|value|value.operation()!=index || value.geometry_source().plan().selections.len()!=2)
                    || (operation.evidence==eredu_core::intervention::InterventionEvidence::None)!=companion.is_none() {
                    return Err(ConstructionError::Invalid("logical evidence companion differs from operation"));
                }
                Ok(EvidenceRows::Borrowed(companion))
            }
        }
    }
}
enum EvidenceRows<'a> {
    Owned(Vec<(CaptureSelection,eredu_core::ObservationPoint)>),
    Borrowed(Option<&'a InterventionEvidenceCompanion>),
}
impl EvidenceRows<'_> {
    fn len(&self)->usize {match self {Self::Owned(rows)=>rows.len(),Self::Borrowed(value)=>value.map_or(0,|value|value.geometry_source().plan().selections.len())}}
    fn get(&self,index:usize)->(&CaptureSelection,&eredu_core::ObservationPoint) {
        match self {
            Self::Owned(rows)=>{let (selection,point)=&rows[index];(selection,point)},
            Self::Borrowed(Some(value))=>(&value.geometry_source().plan().selections[index],&value.geometry_source().points()[index]),
            Self::Borrowed(None)=>unreachable!("empty companion iteration"),
        }
    }
}
struct Lane {
    payloads: Vec<State>,
    any_applied: bool,
    pending_applied: bool,
    pending_affected: u64,
    sealed: bool,
}

pub(super) struct Interventions {
    lanes: Vec<Lane>,
}

fn eligible(
    operation: &eredu_core::intervention::InterventionOperation,
    scope: SpeculativeCaptureScope,
    phase: SpeculativeActivationPhase,
    prediction: u64,
) -> bool {
    scope.applies(phase)
        && operation
            .schedule
            .includes(CapturePhase::Prefill, prediction)
}

impl Interventions {
    pub(super) fn empty() -> Self {
        Self { lanes: Vec::new() }
    }
    pub(super) fn count(
        session: &CaptureSession,
        scopes: &[SpeculativeCaptureScope],
        origin: SpeculativeActivationOrigin,
    ) -> Result<usize, CaptureError> {
        let Some(run)=&session.interventions else {return Ok(0);};
        Self::count_source(&run.plan,scopes,origin).map_err(ConstructionError::into_capture)
    }
    fn count_source(plan:&AdmittedInterventionPlan,scopes:&[SpeculativeCaptureScope],origin:SpeculativeActivationOrigin)->Result<usize,ConstructionError> {
        if scopes.len()!=plan.plan().operations.len() || scopes.len()!=plan.points().len() {return Err(ConstructionError::Invalid("logical intervention scopes differ from source"));}
        let prediction=u64::try_from(origin.prediction).map_err(|_|ConstructionError::Overflow)?;
        phases().into_iter().try_fold(0usize,|total,phase| {
            let count=plan.plan().operations.iter().zip(scopes).filter(|(op,scope)|eligible(op,**scope,phase,prediction)).count();
            total.checked_add(count).ok_or(ConstructionError::Overflow)
        })
    }

    pub(super) fn create(
        session: &mut CaptureSession,
        scopes: &[SpeculativeCaptureScope],
        geometry: SpeculativePrefillReductionGeometry,
        origin: SpeculativeActivationOrigin,
        report: &mut SpeculativePrefillReductions,
    ) -> Result<Self, CaptureError> {
        let Some(run)=&session.interventions else {return Ok(Self::empty());};
        Self::create_shared(Source::Ordinary(&run.plan),&mut session.ledger,scopes,geometry,origin,report,Metadata::ordinary())
            .map_err(ConstructionError::into_capture)
    }
    fn create_shared(source:Source<'_>,ledger:&mut CaptureLedger,scopes:&[SpeculativeCaptureScope],geometry:SpeculativePrefillReductionGeometry,origin:SpeculativeActivationOrigin,report:&mut SpeculativePrefillReductions,metadata:Metadata<'_>)->Result<Self,ConstructionError> {
        metadata.controls(Self::construction_control_bytes().ok_or(ConstructionError::Overflow)?)?;
        let plan=source.plan();
        let count = Self::count_source(plan, scopes, origin)?;
        if count == 0 {
            return Ok(Self::empty());
        }
        let controls = CaptureUsage {
            host_bytes: add(
                std::mem::size_of::<Self>() as u64,
                mul(
                    count as u64,
                    (std::mem::size_of::<Entry>() + std::mem::size_of::<Lane>()) as u64,
                )?,
            )?,
            encoded_bytes: mul(count as u64, 512)?,
            ..Default::default()
        };
        ledger.reserve_quota(controls)?;
        report.charged = report.charged.checked_add(controls)?;
        report.interventions = metadata.vec(count)?;
        let mut out = Self {
            lanes: metadata.vec(count)?,
        };
        let prediction = u64::try_from(origin.prediction).map_err(|_| CaptureError::Overflow)?;
        for phase in phases() {
            for (index, ((operation, point), scope)) in plan
                .plan()
                .operations
                .iter()
                .zip(plan.points())
                .zip(scopes)
                .enumerate()
            {
                if !eligible(operation, *scope, phase, prediction) {
                    continue;
                }
                let rows = length(geometry, phase);
                // Reuse the existing operation/evidence declaration worker. Its
                // temporary projection and final record each retain their own
                // metadata allowance; neither repeats the action payload.
                let declaration_usage = crate::intervention::intervention_metadata(
                    operation,
                    point,
                    plan.identity(),
                )?;
                ledger.reserve_quota(declaration_usage)?;
                report.charged = report.charged.checked_add(declaration_usage)?;
                let selections = source.evidence(index)?;
                let evidence_controls = CaptureUsage {
                    host_bytes: mul(
                        selections.len() as u64,
                        (std::mem::size_of::<CaptureRecord>() + std::mem::size_of::<State>())
                            as u64,
                    )?,
                    ..Default::default()
                };
                ledger.reserve_quota(evidence_controls)?;
                report.charged = report.charged.checked_add(evidence_controls)?;
                ledger.reserve_quota(declaration_usage)?;
                report.charged = report.charged.checked_add(declaration_usage)?;
                let mut evidence = metadata.vec(selections.len())?;
                let mut payloads = metadata.vec(selections.len())?;
                for side in 0..selections.len() {
                    let (selection, point)=selections.get(side);
                    let (record, state) = payload::prepare_with_metadata(
                        ledger,
                        &mut report.charged,
                        point,
                        selection,
                        rows,
                        metadata,
                    )?;
                    evidence.push(record);
                    payloads.push(state);
                }
                report.interventions.push(Entry {
                    phase,
                    operation_index: index,
                    logical_sequence: rows,
                    covered_sequence: 0,
                    first_invocation: None,
                    last_invocation: None,
                    windows: 0,
                    status: if rows == 0 {
                        Status::Skipped
                    } else {
                        Status::Pending
                    },
                    record: InterventionRecord {
                        schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
                        plan_id: metadata.text(plan.identity())?,
                        operation_id: metadata.text(&operation.id)?,
                        target: metadata.text(&operation.target)?,
                        node_id: metadata.text(&point.node_id)?,
                        phase: CapturePhase::Prefill,
                        prediction_index: prediction,
                        outcome: if rows == 0 {
                            InterventionOutcome::Inactive
                        } else {
                            InterventionOutcome::Missing
                        },
                        evidence,
                        charged: declaration_usage,
                        routed_units: point.routed_units.as_ref().map(|_| {
                            RoutedUnitInterventionReceipt {
                                source_tokens: rows,
                                completed_tokens: 0,
                                affected_values: 0,
                            }
                        }),
                    },
                });
                out.lanes.push(Lane {
                    payloads,
                    any_applied: false,
                    pending_applied: false,
                    pending_affected: 0,
                    sealed: rows == 0,
                });
            }
        }
        Ok(out)
    }

    pub(super) fn begin(
        &self,
        entries: &[Entry],
        session: &mut CaptureSession,
        phase: SpeculativeActivationPhase,
        start: u64,
        end: u64,
    ) -> Result<(), CaptureError> {
        self.validate_begin_fixed(entries,phase,start,end).map_err(ConstructionError::into_capture)?;
        for entry in entries.iter().filter(|entry| entry.phase == phase) {
            let physical = session
                .interventions
                .as_mut()
                .and_then(|run| run.records.as_mut())
                .and_then(|records| records.get_mut(entry.operation_index))
                .ok_or_else(|| invalid("logical intervention has no physical operation"))?;
            for (logical, physical) in entry.record.evidence.iter().zip(&mut physical.evidence) {
                if let CaptureOutcome::Skipped { reason } = &logical.outcome {
                    physical.outcome = CaptureOutcome::Skipped {
                        reason: reason.clone(),
                    };
                }
            }
        }
        Ok(())
    }

    pub(super) fn validate_begin_fixed(&self,entries:&[Entry],phase:SpeculativeActivationPhase,start:u64,end:u64)->Result<(),ConstructionError> {
        if entries.len()!=self.lanes.len() || start>end {return Err(ConstructionError::Invalid("logical intervention lane population differs"));}
        for entry in entries.iter().filter(|entry|entry.phase==phase) {
            if entry.covered_sequence!=start || end>entry.logical_sequence {return Err(ConstructionError::Invalid("logical intervention windows are not contiguous"));}
        }
        Ok(())
    }
    pub(super) fn prepare_fixed(
        &mut self,
        entries: &mut [Entry],
        records: &[InterventionRecord],
        phase: SpeculativeActivationPhase,
        start: u64,
        end: u64,
    ) -> Result<(), ConstructionError> {
        // All identity, accounting and scalar algebra checks precede updates.
        for (entry, lane) in entries.iter().zip(&mut self.lanes) {
            if entry.phase != phase {
                continue;
            }
            let physical = records
                .get(entry.operation_index)
                .ok_or_else(|| ConstructionError::Invalid("missing physical intervention record"))?;
            if physical.plan_id != entry.record.plan_id
                || physical.operation_id != entry.record.operation_id
                || physical.target != entry.record.target
                || physical.node_id != entry.record.node_id
                || physical.phase != entry.record.phase
                || physical.prediction_index != entry.record.prediction_index
                || physical.evidence.len() != entry.record.evidence.len()
                || !matches!(
                    physical.outcome,
                    InterventionOutcome::Applied | InterventionOutcome::Unmatched
                )
            {
                return Err(ConstructionError::Invalid("logical intervention source or outcome changed"));
            }
            entry.windows.checked_add(1).ok_or(CaptureError::Overflow)?;
            entry.record.charged.checked_add(physical.charged)?;
            lane.pending_applied =
                lane.any_applied || physical.outcome == InterventionOutcome::Applied;
            match (entry.record.routed_units, physical.routed_units) {
                (Some(total), Some(value))
                    if value.source_tokens == end - start
                        && value.completed_tokens == value.source_tokens =>
                {
                    lane.pending_affected = add(total.affected_values, value.affected_values)?;
                }
                (None, None) => (),
                _ => {
                    return Err(ConstructionError::Invalid(
                        "logical sparse receipt has incomplete physical coverage",
                    ))
                }
            }
            for ((logical, physical), state) in entry
                .record
                .evidence
                .iter()
                .zip(&physical.evidence)
                .zip(&mut lane.payloads)
            {
                if logical.selection_id != physical.selection_id
                    || logical.path != physical.path
                    || logical.node_id != physical.node_id
                    || logical.position != physical.position
                {
                    return Err(ConstructionError::Invalid("logical intervention evidence identity changed"));
                }
                logical.charged.checked_add(physical.charged)?;
                if matches!(logical.outcome, CaptureOutcome::Skipped { .. })
                    || matches!(
                        physical.outcome,
                        CaptureOutcome::Skipped {
                            reason: CaptureSkipReason::Limit { .. }
                        }
                    )
                {
                    continue;
                }
                state.validate_fixed(logical, physical, start, end, entry.logical_sequence)?;
            }
        }
        for (entry, lane) in entries.iter_mut().zip(&mut self.lanes) {
            if entry.phase != phase {
                continue;
            }
            let physical = &records[entry.operation_index];
            for ((logical, physical), state) in entry
                .record
                .evidence
                .iter_mut()
                .zip(&physical.evidence)
                .zip(&mut lane.payloads)
            {
                if matches!(logical.outcome, CaptureOutcome::Skipped { .. }) {
                    continue;
                }
                if let CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Limit { .. },
                } = &physical.outcome
                {
                    logical.outcome = physical.outcome.clone();
                    logical.payload = None;
                    continue;
                }
                state.update_fixed(logical, physical, start, end)?;
            }
        }
        Ok(())
    }

    pub(super) fn finish_window(
        &mut self,
        entries: &mut [Entry],
        records: &[InterventionRecord],
        phase: SpeculativeActivationPhase,
        end: u64,
        invocation: u64,
        success: bool,
    ) {
        for (entry, lane) in entries.iter_mut().zip(&mut self.lanes) {
            if entry.phase != phase {
                continue;
            }
            if !success {
                entry.status = Status::Failed;
                continue;
            }
            let physical = &records[entry.operation_index];
            entry.record.charged = entry
                .record
                .charged
                .checked_add(physical.charged)
                .expect("validated operation charge");
            for ((logical, physical), state) in entry
                .record
                .evidence
                .iter_mut()
                .zip(&physical.evidence)
                .zip(&mut lane.payloads)
            {
                logical.charged = logical
                    .charged
                    .checked_add(physical.charged)
                    .expect("validated evidence charge");
                state.commit(end, entry.logical_sequence);
            }
            lane.any_applied = lane.pending_applied;
            entry.record.outcome = if lane.any_applied {
                InterventionOutcome::Applied
            } else {
                InterventionOutcome::Unmatched
            };
            if let Some(total) = &mut entry.record.routed_units {
                total.completed_tokens = end;
                total.affected_values = lane.pending_affected;
            }
            entry.first_invocation.get_or_insert(invocation);
            entry.last_invocation = Some(invocation);
            entry.windows += 1;
            entry.covered_sequence = end;
            lane.sealed = end == entry.logical_sequence;
        }
    }

    pub(super) fn sealed(&self) -> bool {
        self.lanes.iter().all(|lane| lane.sealed)
    }

    pub(super) fn finish(&self, entries: &mut [Entry], success: bool) {
        for (entry, lane) in entries.iter_mut().zip(&self.lanes) {
            if success && lane.sealed && entry.status == Status::Pending {
                entry.status = Status::Complete;
                for record in &mut entry.record.evidence {
                    if !matches!(record.outcome, CaptureOutcome::Skipped { .. }) {
                        record.outcome = crate::capture::encoded::prefill_terminal_outcome(record)
                            .expect("terminal evidence checked at seal");
                    }
                }
            } else {
                if entry.status == Status::Pending {
                    entry.status = if success {
                        Status::Failed
                    } else {
                        Status::Aborted
                    };
                }
                for record in &mut entry.record.evidence {
                    record.payload = None;
                }
            }
        }
    }
}

impl Interventions {
    fn construction_control_bytes()->Option<usize> {
        use std::mem::{size_of,size_of_val};
        let frames=[size_of::<Self>(),size_of::<Source<'static>>(),size_of::<EvidenceRows<'static>>(),size_of::<Lane>(),size_of::<Entry>(),size_of::<State>(),size_of::<CaptureRecord>(),size_of::<CaptureUsage>(),size_of::<Result<Self,ConstructionError>>(),size_of::<Result<usize,ConstructionError>>(),size_of::<Result<EvidenceRows<'static>,ConstructionError>>(),size_of::<(&AdmittedInterventionPlan,&[SpeculativeCaptureScope],SpeculativeActivationOrigin)>(),size_of::<(Source<'static>,&mut CaptureLedger,&[SpeculativeCaptureScope],SpeculativePrefillReductionGeometry,SpeculativeActivationOrigin,&mut SpeculativePrefillReductions,Metadata<'static>)>(),size_of::<(&eredu_core::intervention::InterventionOperation,&eredu_core::intervention::InterventionPoint)>(),size_of::<Vec<Entry>>(),size_of::<Vec<Lane>>(),size_of::<Vec<State>>(),size_of::<Vec<CaptureRecord>>(),size_of::<[SpeculativeActivationPhase;2]>(),size_of::<[u64;3]>(),size_of::<std::ops::Range<usize>>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
impl WindowReductions {
    pub(in crate::capture::speculative) fn count_original_interventions(source:&OriginalInterventionSource,scopes:&[SpeculativeCaptureScope],origin:SpeculativeActivationOrigin)->Result<usize,ConstructionError> {
        Interventions::count_source(source.plan().admission(),scopes,origin)
    }
    pub(in crate::capture::speculative) fn create_original_interventions(&mut self,source:&OriginalInterventionSource,ledger:&mut CaptureLedger,scopes:&[SpeculativeCaptureScope],origin:SpeculativeActivationOrigin,metadata:Metadata<'_>)->Result<(),ConstructionError> {
        if !self.report.interventions.is_empty() || !self.interventions.lanes.is_empty() || self.pending.is_some() {return Err(ConstructionError::Invalid("logical intervention aggregate is already initialized"));}
        self.interventions=Interventions::create_shared(Source::Original(source),ledger,scopes,self.geometry,origin,&mut self.report,metadata)?;
        Ok(())
    }
    pub(in crate::capture::speculative) fn intervention_evidence_skip(&self,phase:SpeculativeActivationPhase,operation:usize,side:InterventionEvidenceSide)->Option<&CaptureSkipReason> {
        let entry=self.report.interventions.iter().find(|entry|entry.phase==phase && entry.operation_index==operation)?;
        match &entry.record.evidence.get(side.index())?.outcome {CaptureOutcome::Skipped{reason}=>Some(reason),_=>None}
    }
}

impl WindowReductions {
    pub(in crate::capture::speculative) fn intervention_access_control_bytes()->Option<usize> {
        use std::mem::{size_of,size_of_val};
        let frames=[size_of::<(&OriginalInterventionSource,&[SpeculativeCaptureScope],SpeculativeActivationOrigin)>(),size_of::<Result<usize,ConstructionError>>(),size_of::<(&mut Self,&OriginalInterventionSource,&mut CaptureLedger,&[SpeculativeCaptureScope],SpeculativeActivationOrigin,Metadata<'static>)>(),size_of::<Result<(),ConstructionError>>(),size_of::<(&Self,SpeculativeActivationPhase,usize,InterventionEvidenceSide)>(),size_of::<Option<&CaptureSkipReason>>(),size_of::<std::slice::Iter<'static,Entry>>(),size_of::<std::slice::Iter<'static,SpeculativeCaptureScope>>(),size_of::<[SpeculativeActivationPhase;2]>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
#[cfg(test)]
mod tests;
