//! The same canonical serialization feeds ordinary and derived admission IDs.
use super::*;

struct Writer(Sha256);
impl std::io::Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

pub(super) fn digest(
    plan: &CapturePlan, points: &[ObservationPoint], request: CaptureRequestShape,
    bounds: Option<CaptureInvocationBounds>, origin: CaptureTextOrigin,
) -> Result<[u8; 64], serde_json::Error> {
    let mut writer = Writer(Sha256::new());
    match bounds {
        Some(bounds) => serde_json::to_writer(&mut writer, &("invocation", plan, points, bounds)),
        None if origin == CaptureTextOrigin::default() => {
            serde_json::to_writer(&mut writer, &(plan, points, request))
        }
        None => serde_json::to_writer(&mut writer, &("text_origin", plan, points, request, origin)),
    }?;
    let mut output = [0; 64];
    for (index, byte) in writer.0.finalize().iter().copied().enumerate() {
        output[index * 2] = b"0123456789abcdef"[usize::from(byte >> 4)];
        output[index * 2 + 1] = b"0123456789abcdef"[usize::from(byte & 15)];
    }
    Ok(output)
}

pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [size_of::<Writer>(), size_of::<[u8;64]>(), size_of::<[u8;32]>(),
        size_of::<serde_json::Serializer<&mut Writer>>(),
        size_of::<(&str,&CapturePlan,&[ObservationPoint],CaptureRequestShape,CaptureTextOrigin)>(),
        size_of::<(&str,&CapturePlan,&[ObservationPoint],CaptureInvocationBounds)>(),
        size_of::<Result<[u8;64],serde_json::Error>>()]
        .into_iter().try_fold(0usize,usize::checked_add)
}
