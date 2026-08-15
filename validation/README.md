# Checkpoint validation tools

The text-checkpoint path uses three artifacts:

1. `checkpoint_probe` runs SafeMLX and writes JSON metadata plus F32 logits in
   SafeTensors.
2. `reference_runner.py` runs Transformers with the probe's exact prompt and
   cache-feed token IDs.
3. `compare_checkpoints.py` checks input identity, finiteness, relative L2,
   cosine similarity, top-k overlap, and unambiguous argmax agreement.

Install the reference dependencies in a CUDA-enabled Python environment:

```bash
python -m pip install -r validation/requirements.txt
```

Run SafeMLX:

```bash
cargo run -p safemlx-lm --features cuda --example checkpoint_probe -- \
  --model /models/tinyllama \
  --input-ids 1,2,3,4 \
  --teacher-forced-ids 5,6,7,8 \
  --output validation/results/tinyllama_safemlx
```

Run the reference along the exact same token path:

```bash
python validation/reference_runner.py \
  --probe validation/results/tinyllama_safemlx.json \
  --output validation/results/tinyllama_transformers \
  --device cuda:0 --dtype bfloat16
```

Compare using the case's tolerance profile from the manifest:

```bash
python validation/compare_checkpoints.py \
  --actual validation/results/tinyllama_safemlx.json \
  --reference validation/results/tinyllama_transformers.json \
  --manifest validation/models.yaml \
  --case llama_dense_untied \
  --output validation/results/tinyllama_comparison.json
```

The comparator exits `0` on success, `1` for a correctness failure, and `2`
for an invalid invocation or artifact. The current runner targets text inputs
accepted by `AutoModelForCausalLM`; multimodal and realtime families need
their modality-specific reference runners.
