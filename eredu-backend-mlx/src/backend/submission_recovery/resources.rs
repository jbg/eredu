//! A bounded, non-progressing observation of the current thread's recovery list.
use super::*;
use eredu_runtime::detached_resources::{
    DetachedResourceInventory, DetachedResourceLimits, DetachedResourceScan,
};

thread_local! {
    static OBSERVING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn describe_resources(
    limits: DetachedResourceLimits,
) -> Result<DetachedResourceInventory, ResourceDescriptionError> {
    let source = format!("MLX detached submissions on {:?}; active work, ordinary retirement, other threads and unregistered/thread-exit holdings are excluded", std::thread::current().id());
    let unavailable = |reason: &str| {
        DetachedResourceInventory::new(
            source.clone(),
            limits,
            0,
            0,
            DetachedResourceScan::Unavailable {
                reason: reason.into(),
            },
            vec![],
        )
    };
    if REAPING.try_with(|flag| flag.get()).unwrap_or(true) {
        return unavailable(
            "detached retirement is in progress; its list may be temporarily extracted",
        );
    }
    if OBSERVING
        .try_with(|flag| flag.replace(true))
        .unwrap_or(true)
    {
        return unavailable("recursive detached resource observation");
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            let _ = OBSERVING.try_with(|flag| flag.set(false));
        }
    }
    let _reset = Reset;
    ORPHANS
        .try_with(|orphans| {
            let Ok(list) = orphans.try_borrow() else {
                return unavailable(
                    "detached submission list is borrowed; observation does not block or reap",
                );
            };
            let mut node = list.0.as_deref();
            let mut inspected = 0;
            let mut unknown = 0;
            let mut remaining = limits.allocations();
            let mut descriptions = Vec::new();
            while let Some(current) = node {
                if inspected == limits.nodes() {
                    break;
                }
                inspected += 1;
                if let Some(description) = current.describe_resources(remaining)? {
                    remaining = remaining
                        .checked_sub(description.resources.allocations.len())
                        .ok_or_else(|| {
                            ResourceDescriptionError::Invalid(
                                "native detached description exceeded its allowance".into(),
                            )
                        })?;
                    descriptions.push(description);
                } else {
                    unknown += 1;
                }
                node = current.next();
            }
            DetachedResourceInventory::new(
                source.clone(),
                limits,
                inspected,
                unknown,
                if node.is_some() {
                    DetachedResourceScan::Truncated
                } else {
                    DetachedResourceScan::Complete
                },
                descriptions,
            )
        })
        .unwrap_or_else(|_| unavailable("detached submission thread-local storage has exited"))
}

#[cfg(test)]
mod tests;
