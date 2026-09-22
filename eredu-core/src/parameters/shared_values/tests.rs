use super::*;
use crate::{capture::CaptureUsage, intervention::InterventionDtype, parameters::ParameterRegion};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn values() -> ParameterValues {
    ParameterValues {
        identity: "execution:7".into(),
        parameter: "embedding.weight".into(),
        dtype: InterventionDtype::Float32,
        region: ParameterRegion {
            starts: vec![2, 0],
            shape: vec![1, 3],
        },
        values: vec![0.5, -2.0, 7.25],
        usage: CaptureUsage::default(),
    }
}
#[test]
fn parameter_aliases_keep_values_geometry_and_custody_until_the_last_owner() {
    let retired = Arc::new(AtomicUsize::new(0));
    let raw = values();
    let wire = serde_json::to_string(&raw).unwrap();
    let result = SharedParameterValues::retain(raw, Retired(retired.clone()));
    let alias = result.clone();
    assert!(result.same_storage(&alias));
    assert_eq!(result.values.as_ptr(), alias.values.as_ptr());
    assert_eq!(result.region.starts.as_ptr(), alias.region.starts.as_ptr());
    assert_eq!(result.identity.as_ptr(), alias.identity.as_ptr());
    assert_eq!(serde_json::to_string(&result).unwrap(), wire);
    let decoded: SharedParameterValues = serde_json::from_str(&wire).unwrap();
    assert_eq!(result, decoded);
    assert!(!result.same_storage(&decoded));
    drop(result);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(decoded.values, [0.5, -2.0, 7.25]);
}
#[test]
fn projection_aliases_preserve_wire_and_retire_once_across_threads() {
    let retired = Arc::new(AtomicUsize::new(0));
    let raw = ParameterProjectionValues {
        identity: "execution:7".into(),
        parameter: "head.weight".into(),
        source_dtype: InterventionDtype::Float32,
        shape: vec![2, 1],
        values: vec![3.5, -0.25],
        usage: CaptureUsage::default(),
    };
    let wire = serde_json::to_string(&raw).unwrap();
    let result = SharedParameterProjectionValues::retain(raw, Retired(retired.clone()));
    let aliases: Vec<_> = (0..8).map(|_| result.clone()).collect();
    assert!(aliases.iter().all(|alias| result.same_storage(alias)));
    assert_eq!(serde_json::to_string(&result).unwrap(), wire);
    let decoded: SharedParameterProjectionValues = serde_json::from_str(&wire).unwrap();
    assert_eq!(result, decoded);
    drop(result);
    std::thread::scope(|scope| {
        for alias in aliases {
            scope.spawn(move || drop(alias));
        }
    });
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(decoded.values, [3.5, -0.25]);
}
#[test]
fn payload_retires_before_final_custody_even_on_unwind() {
    struct Payload(Arc<AtomicUsize>, bool);
    impl Drop for Payload {
        fn drop(&mut self) {
            self.0.store(1, Ordering::SeqCst);
            if self.1 {
                panic!("payload retirement");
            }
        }
    }
    struct Final(Arc<AtomicUsize>);
    impl Drop for Final {
        fn drop(&mut self) {
            assert_eq!(self.0.swap(2, Ordering::SeqCst), 1);
        }
    }
    for unwind in [false, true] {
        let progress = Arc::new(AtomicUsize::new(0));
        let owner = Owner::retain(Payload(progress.clone(), unwind), Final(progress.clone()));
        let retired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner)));
        assert_eq!(retired.is_err(), unwind);
        assert_eq!(progress.load(Ordering::SeqCst), 2);
    }
    assert!(
        SharedParameterValues::retained_control_bytes::<Retired>().unwrap()
            > std::mem::size_of::<ParameterValues>() as u64
    );
    assert!(
        SharedParameterProjectionValues::retained_control_bytes::<Retired>().unwrap()
            > std::mem::size_of::<ParameterProjectionValues>() as u64
    );
}
