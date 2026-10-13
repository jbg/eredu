//! Pinned upstream GGML scalar-block decoding references.

/// Raw blocks and exact F32 oracle bits from GGML revision
/// `2145525a4081d66ff1a87cf43ef809f95a85ac0c`, seed `0x4e58`.
/// Each pipe-separated row contains a format ID, encoded hex, and decoded hex.
pub const SCALAR_BLOCK_ROWS_REFERENCE: &str =
    include_str!("../fixtures/gguf/scalar-block-rows.txt");
