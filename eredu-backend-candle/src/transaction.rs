use std::{any::Any, rc::Rc};

use eredu_core::{
    backend::{TextSubmissionScope, TextTransactionError, TextTransactionMechanisms},
    eager_completion::EagerCompletion,
    session_authority::{SessionAuthorityError, SessionRetirement},
    BackendFailure, BackendFailureKind, Submission, SubmissionLease,
};

use crate::{CandleContext, CandleTensor};

/// Submission authority for one synchronously settled Candle text invocation.
pub struct CandleTextScope {
    context: CandleContext,
    lease: SubmissionLease,
}

impl CandleTextScope {
    fn settle_scope(
        self,
        root: Rc<dyn Any>,
        output: Option<CandleTensor>,
    ) -> Result<SubmissionLease, BackendFailure> {
        crate::device::CandleSettlement::new(&self.context, (root, output, self.lease))
            .finish()
            .map(|(_, _, lease)| lease)
    }
}

impl TextSubmissionScope<CandleTensor, BackendFailure> for CandleTextScope {
    type Completion = EagerCompletion<BackendFailure, SubmissionLease>;

    fn abandon(self, root: Rc<dyn Any>) {
        let _ = self.settle_scope(root, None);
    }

    fn finish(
        self,
        root: Rc<dyn Any>,
        output: Result<CandleTensor, BackendFailure>,
    ) -> Result<Submission<CandleTensor, Self::Completion>, BackendFailure> {
        let lease = self.settle_scope(root, output.as_ref().ok().cloned())?;
        Ok(Submission {
            output: output?,
            completion: EagerCompletion::completed_with_lease(lease),
        })
    }

    fn validate_tokens(&mut self) -> Result<(), BackendFailure> {
        self.context.synchronize()
    }

    fn settle<T>(
        self,
        root: Rc<dyn Any>,
        result: Result<T, BackendFailure>,
    ) -> Result<T, BackendFailure> {
        let _lease = self.settle_scope(root, None)?;
        result
    }
}

impl SessionRetirement for CandleContext {
    type Error = BackendFailure;

    fn refresh_native_session(&self) -> Result<(), Self::Error> {
        Self::reclaim_retired();
        self.check_healthy()
    }

    fn native_session_fenced(&self) -> bool {
        self.fenced()
    }

    fn submission_authority_error(error: SessionAuthorityError) -> Self::Error {
        error.into()
    }
}

impl TextTransactionMechanisms for CandleContext {
    type Scores = CandleTensor;
    type Scope<'a> = CandleTextScope;

    fn open_transaction(&self, lease: SubmissionLease) -> Result<CandleTextScope, Self::Error> {
        Self::reclaim_retired();
        self.check_healthy()?;
        Ok(CandleTextScope {
            context: self.clone(),
            lease,
        })
    }

    fn transaction_error(error: TextTransactionError) -> Self::Error {
        let kind = match error {
            TextTransactionError::Busy => BackendFailureKind::Busy,
            TextTransactionError::AbsentAuthority => BackendFailureKind::InvalidSession,
        };
        BackendFailure::new(kind, error)
    }
}
