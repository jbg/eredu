#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReferencePayloadRead {
    pub(crate) task: String,
    pub(crate) source: String,
    pub(crate) selection: TensorSelection,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) encoded_bytes: u64,
    pub(crate) physically_bounded: bool,
}
