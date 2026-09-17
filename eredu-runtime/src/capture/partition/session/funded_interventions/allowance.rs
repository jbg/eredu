//! Short original local quota loan; native authority remains with the frame claim.
use super::*;
/// A prepaid local edit loan issued only after the original member preflight.
/// It cannot finish a global claim, authorize native work, or refund its parent.
#[derive(Debug)]
#[must_use = "return the original local loan to its member vote after execution"]
pub struct PartitionInterventionLocalAllowance {
    pub(super) member: Member,
    pub(super) quota: CaptureQuota,
    pub(super) window: Option<InterventionPrefillWindow>,
    pub(super) index: usize,
    pub(super) epoch: DistributedCommitEpoch,
    pub(super) operation: usize,
    pub(super) phase: CapturePhase,
    pub(super) prediction: u64,
    pub(super) identity: Arc<Identity>,
    pub(super) source: OriginalInterventionSource,
    pub(super) metadata: WorkspaceMetadataFunding,
    pub(super) spent: bool,
    pub(super) charges: u8,
    pub(super) validated: bool,
}
impl PartitionInterventionLocalAllowance {
    /// Validate the actual original claim and retained local program, not merely
    /// equal byte counts or shape. Native scope checks remain mandatory as well.
    pub fn validate(
        &mut self,
        claim: &CaptureInterventionClaim<'_>,
        projection: &PreparedPartitionInterventionProjection,
        window: Option<InterventionPrefillWindow>,
        execution_identity: [u8; 32],
        shape: &[u64],
        usage: CaptureUsage,
        projection_usage: [CaptureUsage; 2],
        source_usage: CaptureUsage,
    ) -> Result<(), PartitionInterventionSourceError> {
        let result = (|| -> Result<(), Cause> {
            if self.validated || self.charges != 0 {
                return Err(Cause::Source("local source validation is repeated"));
            }
            claim.validate_source(&self.source)?;
            if claim.index() != self.operation
                || claim.coordinate() != (self.phase, self.prediction)
                || claim.invocation().is_some()
                || window != self.window
                || !projection.source().same_source(&self.source)
                || projection.coordinate() != (self.operation, self.phase, self.prediction, None)
                || projection.geometry_identity() != self.member.geometry
                || execution_identity != self.member.execution
                || shape != self.member.shape
                || usage != self.member.usage
                || projection_usage != self.member.projection
                || source_usage != self.member.source_usage
            {
                return Err(Cause::Source(
                    "actual claim or native program differs from its prepaid local source",
                ));
            }
            if let Some(window) = window {
                window.validate(claim.admission())?;
            }
            self.validated = true;
            Ok(())
        })();
        result.map_err(|cause| self.error(cause))
    }
    pub fn rank(&self) -> usize {
        self.member.rank
    }
    pub fn window(&self) -> Option<InterventionPrefillWindow> {
        self.window
    }
    pub fn reserved(&self) -> CaptureUsage {
        self.quota.limit()
    }
    /// Consume the exact retained dependency completion allowance once, before
    /// completing actual input/output roots through the ordinary native worker.
    pub fn charge_source(
        &mut self,
        usage: CaptureUsage,
    ) -> Result<(), PartitionInterventionSourceError> {
        self.charge(1, usage, self.member.source_usage)
    }
    /// Consume the component then window projection source allowance once.
    pub fn charge_projection(
        &mut self,
        usage: [CaptureUsage; 2],
    ) -> Result<(), PartitionInterventionSourceError> {
        if usage != self.member.projection {
            return Err(self.error(Cause::Source("projection source allowance differs")));
        }
        let expected = usage[0]
            .checked_add(usage[1])
            .map_err(|cause| self.error(cause.into()))?;
        self.charge(2, expected, expected)
    }
    /// Consume only the original local numerical edit estimate, never a parent
    /// ledger or a replacement native submission authority.
    pub fn charge_execution(
        &mut self,
        usage: CaptureUsage,
    ) -> Result<(), PartitionInterventionSourceError> {
        self.charge(4, usage, self.member.usage)
    }
    fn charge(
        &mut self,
        bit: u8,
        actual: CaptureUsage,
        expected: CaptureUsage,
    ) -> Result<(), PartitionInterventionSourceError> {
        if !self.validated || actual != expected || self.charges & bit != 0 {
            return Err(self.error(Cause::Source("local allowance role is repeated or differs")));
        }
        self.quota
            .reserve_quota(actual)
            .map_err(|cause| self.error(cause.into()))?;
        self.charges |= bit;
        self.spent = self.charges == 7;
        Ok(())
    }
    fn error(&self, cause: Cause) -> PartitionInterventionSourceError {
        PartitionInterventionSourceError {
            cause,
            _source: self.source.clone(),
            _metadata: self.metadata.clone(),
        }
    }
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>() * 2,
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), PartitionInterventionSourceError>>(),
            size_of::<PartitionInterventionSourceError>(),
            size_of::<(
                &Self,
                &CaptureInterventionClaim<'_>,
                &PreparedPartitionInterventionProjection,
                Option<InterventionPrefillWindow>,
                [u8; 32],
                &[u64],
                CaptureUsage,
                [CaptureUsage; 2],
                CaptureUsage,
            )>(),
            size_of::<(&mut Self, u8, CaptureUsage, CaptureUsage)>(),
            size_of::<CaptureQuota>(),
            size_of::<[CaptureUsage; 2]>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
