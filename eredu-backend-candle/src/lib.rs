//! Native tensor, neural operator, and submission mechanisms backed by Candle.

mod attention;
mod backend;
mod block_projection;
mod bounded_quantization;
mod cache;
mod capture;
mod checkpoint;
mod component;
mod device;
mod factory;
mod float_cpu;
mod grouped;
mod input_transfer;
mod intervention;
mod matmul;
mod memory;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_bf16;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_f32;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_quantized;
mod operators;
mod prediction;
mod prediction_binding;
mod quantized;
mod realtime;
mod recurrent;
mod residency;
mod rows;
mod sampling;
mod spectral;
mod tensor;
mod text;
mod text_native;
mod transaction;
mod word_projection;

pub use backend::CandleNeuralBackend;
pub use cache::{CandleCacheBuffer, CandleCacheDevice, CandleCachePrefix, CandleCacheStorage};
pub use checkpoint::ParameterReadError;
pub use device::{CandleContext, CandlePlacement};
pub use factory::CandleBackendFactory;
pub use operators::{CandleEmbedding, CandleLinear, CandleNormalization, CandleRotary};
pub use tensor::CandleTensor;
pub use text::{CandlePreparation, CandleToken};
pub use text_native::{CandleInputInspector, CandleObserver, CandleTextValues};
pub use transaction::CandleTextScope;
