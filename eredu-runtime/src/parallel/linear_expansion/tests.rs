use super::*;
use eredu_nn::{Error, workspace::*};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("metadata expansion cannot inspect tensors")
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("metadata only")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        unreachable!()
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("metadata only")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        unreachable!()
    }
}
#[derive(Debug)]
struct Account(Arc<AtomicUsize>, usize);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.0.fetch_add(1, Ordering::SeqCst) == self.1 {
            Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            })
        } else {
            Ok(())
        }
    }
}
fn context(cut: usize) -> (WorkspaceContext, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let funding = HostMetadataFunding::new(Account(calls.clone(), cut)).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap(),
        calls,
    )
}
fn source() -> Vec<ParameterGroupSpec> {
    vec![
        ParameterGroupSpec::partitioned(
            "fused",
            ParameterRole::FeedForwardIntermediate,
            2,
            [ParameterMemberSpec::new(
                "weight",
                [256, 256],
                MemberSharding::PartitionedSegments {
                    axis: 0,
                    segments: vec![0..128, 128..256],
                },
            )],
        )
        .unwrap(),
    ]
}
fn formats() -> Vec<LinearFormatSpec> {
    use eredu_checkpoint::{AffineQuantization, BlockFp8Format, BlockFp8ScaleEncoding};
    let scale = eredu_nn::ParameterSpec::trainable("scale").unwrap();
    let bias = eredu_nn::ParameterSpec::trainable("bias").unwrap();
    vec![
        LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        LinearFormatSpec::affine(
            LinearFormat::Affine(AffineQuantization::new(64, 4).unwrap()),
            scale.clone(),
            bias,
        )
        .unwrap(),
        LinearFormatSpec::scaled(LinearFormat::MxFp4, scale.clone()).unwrap(),
        LinearFormatSpec::scaled(
            LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
            ),
            scale,
        )
        .unwrap(),
        LinearFormatSpec::unscaled(LinearFormat::GgufIQuant {
            ggml_type: eredu_gguf::GgmlType::Q4_0,
            endian: eredu_gguf::Endian::Little,
        })
        .unwrap(),
    ]
}
fn copy_format(
    source: &LinearFormatSpec,
    context: &WorkspaceContext,
) -> Result<Option<LinearFormatSpec>, Error> {
    Ok(Some(LinearFormatSpec::from_parts_with_metadata(
        source.encoding(),
        source
            .scale()
            .map(|v| v.clone_with_metadata(context))
            .transpose()?,
        source
            .affine_bias()
            .map(|v| v.clone_with_metadata(context))
            .transpose()?,
        context,
    )?))
}
#[test]
fn physical_parameter_expansion_preserves_formats_and_stops_at_every_reached_refusal() {
    for format in formats() {
        let expected = ordinary(source(), |_| Ok(Some(format.clone()))).unwrap();
        let (paid, calls) = context(usize::MAX);
        let first = calls.load(Ordering::SeqCst);
        let actual = expand_linear_format_parameter_groups_with_metadata(
            source(),
            |_| copy_format(&format, &paid),
            &paid,
        )
        .unwrap();
        assert_eq!(actual, expected);
        let count = calls.load(Ordering::SeqCst) - first;
        assert!(count > 8);
        for cut in 0..count {
            let (paid, calls) = context(first + cut);
            let result = expand_linear_format_parameter_groups_with_metadata(
                source(),
                |_| copy_format(&format, &paid),
                &paid,
            );
            assert!(result.is_err(), "{:?} cut {cut}/{count}", format.encoding());
            assert_eq!(calls.load(Ordering::SeqCst), first + cut + 1);
        }
    }
}
