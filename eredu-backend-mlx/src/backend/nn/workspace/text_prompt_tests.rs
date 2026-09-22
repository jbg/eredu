use super::*;
use eredu_core::{InferenceGeometry, OutputDemand, TextGenerationBackend};
use safemlx::{ops::indexing::TryIndexOp, Device, DeviceType, Stream};

#[test]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_text_prompt_preparation_prices_complete_backing_across_chunks() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let backend = crate::native::backend(&stream, &stream);
    let facts = MlxMetalWorkspaceMechanisms::current_host()
        .unwrap()
        .ordinary_storage();
    for positions in [1, 37, 3000, 4097, 16385] {
        for spare_capacity in [0, 8192] {
            let mut ids = Vec::with_capacity(positions + spare_capacity);
            ids.extend((0..positions).map(|position| (position % 71 + 1) as u32));
            let host_bytes = ids.capacity() as u64 * 4;
            let identity_peak =
                eredu_runtime::input::TextInputIdentityPlan::new(1, positions as u64)
                    .unwrap()
                    .peak_bytes();
            let fingerprint = eredu_core::cache::prompt_cache_token_fingerprint(&ids);
            let mut quotes = Vec::new();
            stream.synchronize().unwrap();
            let cold_before = safemlx::memory::active_memory().unwrap();
            for chunk in [1, positions.min(128), positions] {
                let quote = eredu_runtime::working_memory::quote_text_prompt_workspace(
                    InferenceGeometry {
                        batch_size: 1,
                        cached_positions: 0,
                        input_positions: positions as u64,
                        max_output_tokens: 3,
                        prefill_chunk_positions: chunk as u64,
                        output: OutputDemand::LastPosition,
                    },
                    Some(host_bytes),
                    &WorkspaceContext::new(facts),
                )
                .unwrap();
                let host = quote.host_peak_bytes().unwrap();
                assert!(
                    host >= host_bytes
                        + identity_peak
                        + facts.allocation().host_control_bytes().unwrap()
                );
                assert_eq!(
                    quote.peak().bytes(),
                    host.checked_add(quote.tensor_peak_bytes().unwrap())
                );
                let minimal = eredu_runtime::working_memory::quote_text_prompt_workspace(
                    quote.geometry(),
                    Some(positions as u64 * 4),
                    &WorkspaceContext::new(facts),
                )
                .unwrap();
                // Changing caller capacity changes exactly that retained host
                // contribution; native and identity controls remain present.
                assert_eq!(
                    host.checked_sub(minimal.host_peak_bytes().unwrap()),
                    host_bytes.checked_sub(positions as u64 * 4)
                );
                quotes.push(quote);
            }
            assert_eq!(safemlx::memory::active_memory().unwrap(), cold_before);
            assert!(quotes
                .windows(2)
                .all(|pair| pair[0].peak().bytes() == pair[1].peak().bytes()));
            safemlx::memory::reset_peak_memory().unwrap();
            let prompt = TextGenerationBackend::prepare_text_prompt(&backend, ids).unwrap();
            let tokens = prompt.with_borrowed(|input| {
                crate::backend::runtime::media::input::text_token_ids(input, &stream).unwrap()
            });
            tokens.evaluated().unwrap();
            stream.synchronize().unwrap();
            let observed = safemlx::memory::peak_memory()
                .unwrap()
                .checked_sub(cold_before)
                .unwrap() as u64;
            let quote = &quotes[0];
            assert!(observed <= quote.tensor_peak_bytes().unwrap());
            assert_eq!(tokens.shape(), [1, positions as i32]);
            assert_eq!(
                eredu_core::cache::prompt_cache_token_fingerprint(
                    tokens.evaluated().unwrap().try_as_slice::<u32>().unwrap()
                ),
                fingerprint
            );
            let identity = prompt.cache_identity().unwrap().clone();
            let backing = tokens.allocation_info().unwrap().unwrap();
            let last = tokens
                .try_index_device((.., positions as i32 - 1..), &stream)
                .unwrap();
            last.evaluated().unwrap();
            assert_eq!(last.allocation_info().unwrap(), Some(backing));
            assert!(backing.bytes() >= positions * 4);
            assert_eq!(prompt.cache_identity(), Some(&identity));
            drop(tokens);
            drop(prompt);
            assert_eq!(last.allocation_info().unwrap(), Some(backing));
            eprintln!(
                "text prompt positions={positions} host_capacity={host_bytes} observed={observed} tensor_bound={} managed_bound={} retained_backing={}",
                quote.tensor_peak_bytes().unwrap(),
                quote.peak().bytes().unwrap(),
                backing.bytes()
            );
        }
    }
}
