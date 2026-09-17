use super::*;
use crate::composition::NeutralActivationObserver;
use eredu_runtime::{prefill::PrefillChunk, ActivationObserver};
use std::{fmt, sync::Arc};

#[derive(Debug)]
struct Original(Arc<()>);
impl fmt::Display for Original {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("original prefill annotation failure")
    }
}
impl std::error::Error for Original {}
struct Observer {
    chunks: Vec<usize>,
    terminals: Vec<bool>,
    failure: Option<Arc<()>>,
}
impl ActivationObserver<MlxTensor, Error> for Observer {
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        self.chunks.push(std::ptr::from_ref(chunk) as usize);
        match &self.failure {
            Some(identity) => Err(Error::Other(Box::new(Original(identity.clone())))),
            None => Ok(()),
        }
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.terminals.push(committed);
    }
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        panic!("lifecycle forwarding must not perform tensor work")
    }
}

#[test]
fn native_observer_adapters_forward_exact_prefill_chunk_and_original_failure() {
    // No Array, stream, allocation authority, or native operation is needed for
    // these borrowed lifecycle callbacks across the actual adapter chain.
    let chunk = PrefillChunk {
        input: 1..2,
        position: 8,
        output: eredu_core::OutputDemand::StateOnly,
    };
    for fail in [false, true] {
        let identity = Arc::new(());
        let mut observer = Observer {
            chunks: vec![],
            terminals: vec![],
            failure: fail.then(|| identity.clone()),
        };
        let mut arrays = ArrayObserverAdapter {
            inner: &mut observer,
            routed_path: None,
            routed_invocation_active: false,
            allocation_authority: None,
        };
        let mut neutral = NeutralActivationObserver::new(&mut arrays);
        let result = neutral.begin_prefill_chunk(&chunk);
        if fail {
            let error = result.unwrap_err();
            let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
            let original = loop {
                let current = cause.expect("original typed source");
                if let Some(original) = current.downcast_ref::<Original>() {
                    break original;
                }
                cause = current.source();
            };
            assert!(Arc::ptr_eq(&original.0, &identity));
        } else {
            result.unwrap();
        }
        neutral.finish_prefill(!fail);
        drop(neutral);
        drop(arrays);
        assert_eq!(observer.chunks, [std::ptr::from_ref(&chunk) as usize]);
        assert_eq!(observer.terminals, [!fail]);
    }
}
