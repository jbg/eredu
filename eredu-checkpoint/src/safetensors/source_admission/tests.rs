use super::*;
use crate::{
    safetensors::{SafetensorsDiscoveryLimits, SafetensorsShardError},
    store::{SafetensorsWeightStore, StoreError},
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
#[derive(Debug, thiserror::Error)]
#[error("index policy refused")]
struct Refused;
#[derive(Debug)]
struct Policy {
    index: PathBuf,
    mode: u8,
    calls: AtomicUsize,
}
impl SafetensorsSourceAdmission for Policy {
    fn reserve_index(
        &self,
        request: SafetensorsIndexRequest,
    ) -> Result<(), Arc<dyn Error + Send + Sync>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(request.encoded_bytes > 0);
        assert!(request.path_bytes > 0);
        match self.mode {
            0 => return Err(Arc::new(Refused)),
            1 => std::fs::write(&self.index, vec![b' '; request.encoded_bytes + 1]).unwrap(),
            2 => std::fs::write(&self.index, vec![b' '; request.encoded_bytes]).unwrap(),
            3 => std::fs::write(&self.index, b"{").unwrap(),
            _ => unreachable!(),
        }
        let file = std::fs::File::options()
            .write(true)
            .open(&self.index)
            .unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(2))
            .unwrap();
        Ok(())
    }
    fn headers(
        &self,
        _: usize,
    ) -> Result<Arc<dyn SafetensorsHeaderAdmission>, Arc<dyn Error + Send + Sync>> {
        panic!("refusal or version check must precede index parsing")
    }
    fn reserve_store(&self, _: usize) -> Result<(), Arc<dyn Error + Send + Sync>> {
        panic!("no store construction")
    }
}
#[test]
fn index_admission_refusal_and_mutations_precede_decoding() {
    for mode in 0..4 {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index, b"malformed input").unwrap();
        let policy = Arc::new(Policy {
            index,
            mode,
            calls: AtomicUsize::new(0),
        });
        let error = SafetensorsWeightStore::open_with_source_admission(
            dir.path(),
            1,
            SafetensorsDiscoveryLimits::default(),
            policy.clone(),
        )
        .unwrap_err();
        assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
        if mode == 0 {
            let StoreError::SafetensorsSourceAdmission(cause) = error else {
                panic!("typed refusal")
            };
            assert!(cause.is::<Refused>());
        } else {
            assert!(
                matches!(
                    error,
                    StoreError::SafetensorsShards(SafetensorsShardError::IndexChanged { .. })
                ),
                "{error:?}"
            );
        }
    }
}
