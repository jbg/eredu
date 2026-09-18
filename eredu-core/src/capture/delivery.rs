/// Another prediction requires draining the preceding capture transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("text capture has undrained records or a pending transaction")]
pub struct CaptureDeliveryPending;
