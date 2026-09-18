# Released Qwen readout and chunk measurements

Historical numerical and allocation evidence is retained here with its original
commands, dates and tolerances. It does not establish current tool admission,
process-wide bounds or the latest controlled/media API. See
[bounded inference](bounded-inference.md) for the current contract.

## Released Qwen readout measurements (2026-09-13)

The [recorded measurements](validation/bounded-qwen-readout-2026-09-13.json)
compare fresh native processes with identical token IDs and three teacher-forced
cached decodes. `last` uses ordinary prefill; `sequence` uses an empty capture
selection that preserves the sequence observer's full-row demand. Both return
one vocabulary row. No hidden captures are retained. Each run processes the
whole prompt in one invocation; these results do not measure native chunking.

| Prompt positions | MLX live peak, last / sequence (GiB) | Process peak footprint, last / sequence (GiB) |
| ---: | ---: | ---: |
| 5 | 3.510 / 3.510 | 5.005 / 5.010 |
| 128 | 4.232 / 4.232 | 5.738 / 5.856 |
| 512 | 6.275 / 6.275 | 7.804 / 8.277 |
| 1,500 | 11.793 / 11.793 | 13.327 / 14.714 |
| 3,000 | 20.893 / 20.893 | 22.558 / 25.331 |

At 3,000 positions, peak process footprint drops by 2,978,283,640 bytes
(2.774 GiB), from 27,199,392,672 to 24,221,109,032 bytes. The observed output dtype
is F32, so a full 3,000 × 248,320 score tensor occupies about 2.775 GiB before
selection. MLX's live peak remains 22,433,750,408 bytes in both modes: other
full-prompt transients establish that earlier peak. This directly demonstrates
why readout selection alone does not finish the working-memory task. MLX cached
allocations, process RSS and process physical footprint are distinct metrics;
the record retains both maximum RSS and peak footprint from `/usr/bin/time -l`.
Neither allocator telemetry nor these measurements establish an enforceable bound.

Across the five lengths, prefill score differences between native readout modes
are at most `2.575e-5`; all three decode score rows match exactly and every argmax
matches. Independent MLX-LM comparison checks every vocabulary score at prompt
lengths 5 and 3,000 plus three cached decodes, with `atol=0.25`, `rtol=0.02`.
Every row and argmax passes. Maximum absolute reference error is `0.210024`;
maximum RMSE is `0.038212`. Native scores are F32, while the independent model
loads the released BF16 weights. Timing includes cold native/JIT effects;
these runs are memory/numerical validation, not a throughput comparison.

Reproduce the reference and one native case with Metal access:

```sh
python3 eredu-backend-mlx/validation/fetch_bounded_qwen_checkpoint.py /private/tmp/qwen-reference
python3 -m venv /private/tmp/qwen-reference-env
/private/tmp/qwen-reference-env/bin/pip install mlx==0.32.2 mlx-lm==0.31.3 transformers==5.17.0
cargo rustc -p eredu-backend-mlx --example readout_memory_probe --features metal,accelerate -- -C link-arg=-Wl,-S
python3 - <<'PYINPUT'
import json
prefix = [760, 6511, 314, 9338, 369]
for positions in [5, 128, 512, 1500, 3000]:
    with open(f'/private/tmp/qwen-input-{positions}.json', 'w') as output:
        json.dump({'prompt_ids': (prefix * ((positions + 4) // 5))[:positions],
                   'decode_ids': [11751, 13, 198]}, output)
PYINPUT
/usr/bin/time -l target/debug/examples/readout_memory_probe /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json last /private/tmp/qwen-last.json 3000
/usr/bin/time -l target/debug/examples/readout_memory_probe /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json sequence /private/tmp/qwen-sequence.json 3000
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /private/tmp/qwen-reference-env/bin/python eredu-backend-mlx/validation/bounded_qwen_reference.py /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json /private/tmp/qwen-reference-scores.json --compare /private/tmp/qwen-last.json
```

Repeat both fresh native processes with each input length for the table. The link
flag strips debug symbols from the probe executable to limit build-disk use; it
does not change numerical compilation settings. A failed initial debug link
exhausted local disk space; deleting generated incremental build caches allowed
the probe to build. The native linker still reports its existing large unwind
section warning. No hardware limitation prevented this single-device matrix.

## Released Qwen chunk measurements (2026-09-13)

The [chunk-policy record](validation/bounded-qwen-chunks-2026-09-13.json) uses
the same checkpoint, hardware and inputs. All 15 runs use final-position readout
and three teacher-forced cached decodes in fresh native processes. `full` sets
the chunk limit to the prompt length; the other policies cap it at 128 or 512.
Both native direct and composite token-only adapters enter the runtime's shared
driver; this released conditional Qwen checkpoint exercises composite ingress.

| Prompt positions | MLX peak, full / 128 / 512 (GiB) | Process peak footprint, full / 128 / 512 (GiB) |
| ---: | ---: | ---: |
| 5 | 3.510 / 3.510 / 3.510 | 5.006 / 5.005 / 5.006 |
| 128 | 4.232 / 4.232 / 4.232 | 5.739 / 5.739 / 5.739 |
| 512 | 6.275 / 4.316 / 6.275 | 7.804 / 5.859 / 7.805 |
| 1,500 | 11.793 / 4.377 / 6.565 | 13.328 / 6.632 / 10.554 |
| 3,000 | 20.893 / 4.483 / 6.775 | 22.559 / 7.721 / 11.332 |

At 3,000 positions, the default 512-position policy reduces measured MLX peak
from 22,433,750,408 to 7,274,563,696 bytes (67.6%) and process peak footprint
from 24,222,780,344 to 12,168,154,064 bytes (49.8%). A 128-position limit further
reduces these peaks to 4,813,825,208 and 8,290,094,032 bytes. These include existing
model residency. Process footprint still grows with prompt length even at a
fixed chunk limit; chunk scheduling alone does not prove total memory bounded.

Every vocabulary score in all four output rows passes native chunk/full
comparison with `atol=1e-4`, `rtol=1e-4`. Maximum absolute difference is
`3.2783e-5`; all argmax results match. Both 3,000-position chunk policies also pass
the independent MLX-LM comparison with the previously declared `atol=0.25`,
`rtol=0.02`. Maximum absolute reference error is `0.210024`, maximum RMSE is
`0.038212`, and all four argmax results match. Native outputs are F32; the
independent implementation loads the released BF16 weights. These tolerances
describe numerical comparison, not a memory-budget guarantee.

After the setup commands above, reproduce each fresh process as follows:

```sh
for n in 5 128 512 1500 3000; do
  for policy in full 128 512; do
    chunk="$policy"
    if [ "$policy" = full ]; then chunk="$n"; fi
    /usr/bin/time -l target/debug/examples/readout_memory_probe \
      /private/tmp/qwen-reference /private/tmp/qwen-input-"$n".json last \
      /private/tmp/qwen-chunk-"$n"-"$policy".json "$chunk" \
      > /private/tmp/qwen-chunk-"$n"-"$policy".stdout \
      2> /private/tmp/qwen-chunk-"$n"-"$policy".time
  done
done
for chunk in 128 512; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /private/tmp/qwen-reference-env/bin/python \
    eredu-backend-mlx/validation/bounded_qwen_reference.py \
    /private/tmp/qwen-reference /private/tmp/qwen-input-3000.json \
    /private/tmp/qwen-chunk-reference-"$chunk".json \
    --compare /private/tmp/qwen-chunk-3000-"$chunk".json
done
```

The native input builder exposes `with_prefill_chunk_positions(NonZeroU64)`;
omitting the probe's final argument selects the default 512-position limit.
These commands describe the recorded September 13 probe and source revision.
They measure the ordinary chunk/readout mechanism, not current public bounded
chat admission. Later managed measurements and their distinct executable hashes
are in [the public Qwen record](validation/bounded-public-qwen-2026-09-16.json).
Current tool/media acceptance is tracked in [the consolidation overview](bounded-followup.md).
