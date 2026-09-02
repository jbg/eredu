#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(all(feature = "metal", target_vendor = "apple"))]
include!(concat!(env!("OUT_DIR"), "/embedded_metallib.rs"));

/// Path to the uncompressed `mlx.metallib` build artifact.
///
/// Normal applications do not need this path: the compressed library is
/// embedded automatically. This artifact remains available for diagnostics
/// and explicit custom-library overrides. Set `SAFEMLX_METALLIB_OUTPUT_DIR`
/// while building to export it to a different directory.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub const MLX_METALLIB_PATH: &str = match option_env!("SAFEMLX_METALLIB_PATH") {
    Some(path) => path,
    None => "",
};
