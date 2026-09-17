//! Constructor-owned native Scope metadata; deliberately unintegrated with funding.
use super::{
    destroy, retire, take_owner, OwnedNode, RetiredOwner, SubmissionGraphQuota,
    SubmissionRecordQuota,
};
use crate::{utils::runtime_lock, SubmissionScope};
use std::{alloc::Layout, ffi::c_void, fmt, mem, ptr};

/// Fixed failure with the unchanged owner returned alongside it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SubmissionScopeOwnerCause {
    /// The Rust preparation node could not be allocated.
    #[error("submission scope owner allocation failed")]
    AllocationFailed,
    /// No runtime loan was available; no native constructor was called.
    #[error("native runtime is busy during owned scope construction")]
    RuntimeBusy,
    /// Native construction failed before consuming the prepared owner.
    #[error("native owned scope construction failed")]
    NativeConstructionFailed,
    /// Default nesting attempted to replace the parent's Record arena.
    #[error("nested submission scope has a foreign Record arena")]
    ForeignRecordArena,
    /// Default nesting attempted to replace the parent's Graph arena.
    #[error("nested submission scope has a foreign Graph arena")]
    ForeignGraphArena,
    /// The explicit child observer is not the configured active parent.
    #[error("original child scope requires its exact active parent outside native dispatch")]
    InvalidOriginalParent,
}

/// Cause first, unchanged owner last. No failure implies native completion.
pub struct SubmissionScopeOwnerError<O> {
    cause: SubmissionScopeOwnerCause,
    owner: O,
}
impl<O> SubmissionScopeOwnerError<O> {
    /// The fixed failure without allocating or consuming its owner.
    pub fn cause(&self) -> SubmissionScopeOwnerCause {
        self.cause
    }
    /// The entire original owner or prepared capsule.
    pub fn owner(&self) -> &O {
        &self.owner
    }
    /// Consume the error while preserving the original returned owner.
    pub fn into_parts(self) -> (SubmissionScopeOwnerCause, O) {
        (self.cause, self.owner)
    }
}
impl<O> fmt::Debug for SubmissionScopeOwnerError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionScopeOwnerError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<O> fmt::Display for SubmissionScopeOwnerError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<O> std::error::Error for SubmissionScopeOwnerError<O> {}

/// Exact named representation facts for one `T` and one native Scope.
///
/// These are diagnostics, not funding or a total memory bound. Payload-owned
/// allocations, allocator overhead, registry/thread initialization, Record
/// storage, transitive native graphs, queue population and compiler-generated
/// stack spills are excluded. Construction/extraction values overlap their
/// result/error wrappers; callers must price an actual finite schedule before
/// admission. The Scope representation includes its new inline callback pair
/// even for ordinary unowned scopes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionScopeOwnerLayout {
    /// One allocated Rust queue node including T.
    pub rust_node_bytes: usize,
    /// One native Scope allocation, with its actual inline callback fields.
    pub native_scope_bytes: usize,
    /// Actual named native final-release tuple (parent, payload, callback).
    pub native_retirement_control_bytes: usize,
    /// The closed, exclusive Rust preparation.
    pub prepared_bytes: usize,
    /// T argument, constructed node value, allocation Layout and pointer locals.
    pub preparation_control_bytes: usize,
    /// Begin locals, constructor function pointer, runtime loan and final result.
    /// Retain this alongside the incoming prepared wrapper during handoff.
    pub begin_control_bytes: usize,
    /// Rust node Box/extraction arguments, returned T and queue batch controls.
    pub retirement_control_bytes: usize,
    /// Failed preparation retains the actual original T inline.
    pub preparation_failure_bytes: usize,
    /// Busy/native rejection retains the actual complete preparation inline.
    pub begin_failure_bytes: usize,
}
impl SubmissionScopeOwnerLayout {
    /// Checked sum of the two allocated representations, without other costs.
    pub fn allocation_bytes(self) -> Option<usize> {
        self.rust_node_bytes.checked_add(self.native_scope_bytes)
    }
}

// The exact temporary used by begin; no native owner escapes this module except
// the completed SubmissionScope raw value returned to its closed constructor.
struct BeginControls {
    raw: safemlx_sys::mlx_submission_scope,
    status: i32,
}
type Constructor = unsafe extern "C" fn(
    *mut safemlx_sys::mlx_submission_scope,
    *mut c_void,
    Option<unsafe extern "C" fn(*mut c_void)>,
) -> i32;

/// One preallocated owner for a new native Scope, not an after-begin attachment.
///
/// Preparation invokes no native code. Failure returns T unchanged. Success
/// keeps the node closed until `SubmissionScope::try_begin_retaining` consumes
/// it; Busy/native failure returns this same capsule for retry or destruction.
/// Both unattached and queued retirement deallocate its Box before dropping T.
/// No raw queue node or native owner handle escapes. T: Send permits the existing
/// global queue to reclaim T on another ordinary host thread; the capsule itself need not move
/// across threads. There is no unsafe Send implementation.
///
/// T must not retain the future Scope, a descendant or any Record retaining it.
/// This primitive neither admits nor prices T, native work or a Scope population.
pub struct PreparedSubmissionScopeOwner<T: Send + 'static> {
    node: Option<Box<OwnedNode<T>>>,
    record_quota: Option<SubmissionRecordQuota>,
    graph_quota: Option<SubmissionGraphQuota>,
}
impl<T: Send + 'static> fmt::Debug for PreparedSubmissionScopeOwner<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSubmissionScopeOwner")
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedSubmissionScopeOwner<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedSubmissionScopeOwner<T> {
    /// Allocate one fixed Rust node; never enters the runtime or runs housekeeping.
    pub fn try_new(owner: T) -> Result<Self, SubmissionScopeOwnerError<T>> {
        // SAFETY: try_new_with initializes the returned allocation immediately
        // and pairs its exact Layout with Box<OwnedNode<T>>.
        Self::try_new_with(owner, |layout| unsafe { std::alloc::alloc(layout) })
    }

    fn try_new_with(
        owner: T,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<Self, SubmissionScopeOwnerError<T>> {
        let layout = Layout::new::<OwnedNode<T>>();
        let node = allocate(layout).cast::<OwnedNode<T>>();
        if node.is_null() {
            return Err(SubmissionScopeOwnerError {
                cause: SubmissionScopeOwnerCause::AllocationFailed,
                owner,
            });
        }
        // SAFETY: the private allocator returns exactly this Layout or null.
        // No fallible/user callback remains before the exclusive Box is owned.
        let node = unsafe {
            node.write(OwnedNode {
                retired: RetiredOwner {
                    next: ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
            Box::from_raw(node)
        };
        Ok(Self {
            node: Some(node),
            record_quota: None,
            graph_quota: None,
        })
    }

    /// Bind the same fixed arena before native construction. Descendant ordinary
    /// scopes inherit it. This does not attach capacity to an existing Scope.
    pub fn with_record_quota(mut self, quota: SubmissionRecordQuota) -> Self {
        self.record_quota = Some(quota);
        self
    }

    /// Bind original graph metadata capacity before native construction. This
    /// arena is independent of the Record arena and cannot refill either one.
    pub fn with_graph_quota(mut self, quota: SubmissionGraphQuota) -> Self {
        self.graph_quota = Some(quota);
        self
    }

    /// Borrow T without exporting its queue node or native owner handle.
    pub fn owner(&self) -> &T {
        &self
            .node
            .as_ref()
            .expect("unconsumed scope preparation")
            .owner
    }

    /// Cold exact type facts; no runtime entry, allocation or housekeeping.
    pub fn layout() -> Option<SubmissionScopeOwnerLayout> {
        fn sum(parts: &[usize]) -> Option<usize> {
            parts.iter().try_fold(0usize, |a, b| a.checked_add(*b))
        }
        Some(SubmissionScopeOwnerLayout {
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            native_scope_bytes: SubmissionScope::native_control_bytes(),
            // SAFETY: native returns sizeof its named non-owning extraction tuple.
            native_retirement_control_bytes: unsafe {
                safemlx_sys::mlx_submission_scope_retirement_control_bytes()
            },
            prepared_bytes: mem::size_of::<Self>(),
            preparation_control_bytes: sum(&[
                mem::size_of::<T>(),
                mem::size_of::<OwnedNode<T>>(),
                mem::size_of::<Layout>(),
                mem::size_of::<*mut OwnedNode<T>>(),
                mem::size_of::<Box<OwnedNode<T>>>(),
            ])?,
            begin_control_bytes: sum(&[
                mem::size_of::<BeginControls>(),
                mem::size_of::<safemlx_sys::mlx_submission_record_quota>(),
                mem::size_of::<safemlx_sys::mlx_submission_graph_quota>(),
                mem::size_of::<Constructor>(),
                mem::size_of::<Option<&crate::OriginalScopeObserver>>(),
                mem::size_of::<safemlx_sys::mlx_submission_observer>(),
                // SAFETY: the native function returns fixed representation facts.
                unsafe { safemlx_sys::mlx_submission_scope_original_child_control_bytes() },
                mem::size_of::<runtime_lock::RuntimeLockGuard>(),
                mem::size_of::<Result<SubmissionScope, SubmissionScopeOwnerError<Self>>>(),
                mem::size_of::<*mut c_void>(),
                mem::size_of::<Box<OwnedNode<T>>>(),
            ])?,
            retirement_control_bytes: sum(&[
                mem::size_of::<Option<Box<OwnedNode<T>>>>(),
                mem::size_of::<Box<OwnedNode<T>>>(),
                mem::size_of::<T>(),
                mem::size_of::<T>(),
                mem::size_of::<super::RetirementBatch>(),
                mem::size_of::<*mut RetiredOwner>(),
                mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            ])?,
            preparation_failure_bytes: mem::size_of::<SubmissionScopeOwnerError<T>>(),
            begin_failure_bytes: mem::size_of::<SubmissionScopeOwnerError<Self>>(),
        })
    }

    pub(crate) fn begin(
        self,
    ) -> Result<safemlx_sys::mlx_submission_scope, SubmissionScopeOwnerError<Self>> {
        self.begin_with(safemlx_sys::mlx_submission_scope_new_retaining)
    }

    pub(crate) fn begin_original_child(
        self,
        parent: &crate::OriginalScopeObserver,
    ) -> Result<safemlx_sys::mlx_submission_scope, SubmissionScopeOwnerError<Self>> {
        self.begin_with_parent(safemlx_sys::mlx_submission_scope_new_retaining, Some(parent))
    }

    fn begin_with(
        self,
        construct: Constructor,
    ) -> Result<safemlx_sys::mlx_submission_scope, SubmissionScopeOwnerError<Self>> {
        self.begin_with_parent(construct, None)
    }

    fn begin_with_parent(
        mut self,
        construct: Constructor,
        parent: Option<&crate::OriginalScopeObserver>,
    ) -> Result<safemlx_sys::mlx_submission_scope, SubmissionScopeOwnerError<Self>> {
        let mut controls = BeginControls {
            raw: safemlx_sys::mlx_submission_scope {
                ctx: ptr::null_mut(),
            },
            status: -1,
        };
        {
            let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
                return Err(SubmissionScopeOwnerError {
                    cause: SubmissionScopeOwnerCause::RuntimeBusy,
                    owner: self,
                });
            };
            let payload = (&mut **self.node.as_mut().expect("unconsumed scope preparation")
                as *mut OwnedNode<T>)
                .cast::<c_void>();
            // SAFETY: this exclusive initialized node stays owned through native
            // construction. The closed constructor consumes it only on status 0,
            // leaves output null on failure, and never calls retire before return.
            controls.status = if let Some(parent) = parent {
                // SAFETY: the borrowed observer pins its actual Scope. Native
                // code validates current-parent identity before owner transfer;
                // absent arenas refuse rather than inheriting unpriced capacity.
                unsafe {
                    safemlx_sys::mlx_submission_scope_new_original_child(
                        &mut controls.raw, payload, Some(retire),
                        self.record_quota.as_ref().map_or(
                            safemlx_sys::mlx_submission_record_quota { ctx: ptr::null_mut() },
                            SubmissionRecordQuota::raw,
                        ),
                        self.graph_quota.as_ref().map_or(
                            safemlx_sys::mlx_submission_graph_quota { ctx: ptr::null_mut() },
                            SubmissionGraphQuota::raw,
                        ),
                        parent.raw,
                    )
                }
            } else if self.record_quota.is_some() || self.graph_quota.is_some() {
                // SAFETY: this closed handle remains live until native success
                // retains it, or the same complete preparation is returned.
                unsafe {
                    safemlx_sys::mlx_submission_scope_new_retaining_with_arenas(
                        &mut controls.raw,
                        payload,
                        Some(retire),
                        self.record_quota.as_ref().map_or(
                            safemlx_sys::mlx_submission_record_quota {
                                ctx: ptr::null_mut(),
                            },
                            SubmissionRecordQuota::raw,
                        ),
                        self.graph_quota.as_ref().map_or(
                            safemlx_sys::mlx_submission_graph_quota {
                                ctx: ptr::null_mut(),
                            },
                            SubmissionGraphQuota::raw,
                        ),
                    )
                }
            } else {
                unsafe { construct(&mut controls.raw, payload, Some(retire)) }
            };
            if controls.status == 0 {
                // Infallible ownership transfer; no allocation or owner destructor.
                let node = self.node.take().expect("unconsumed scope preparation");
                let _ = Box::into_raw(node);
            }
        }
        // The runtime loan is gone before either failure or empty self can drop.
        if controls.status == 0 {
            Ok(controls.raw)
        } else {
            Err(SubmissionScopeOwnerError {
                cause: match controls.status {
                    2 => SubmissionScopeOwnerCause::ForeignRecordArena,
                    3 => SubmissionScopeOwnerCause::ForeignGraphArena,
                    4 => SubmissionScopeOwnerCause::InvalidOriginalParent,
                    _ => SubmissionScopeOwnerCause::NativeConstructionFailed,
                },
                owner: self,
            })
        }
    }
}

#[cfg(test)]
mod tests;
