"""Reproduce pinned GGUF tensor bytes from the official source in bounded rows.

Uses ggml_quantize_chunk from the pinned primary llama.cpp build. The original
publisher's tool revision remains a separate provenance fact.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open
from nemotron_reference_provenance import verify_release


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("artifact",type=Path)
    parser.add_argument("quantizer_library",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--source-precision",choices=("released","f16"),default="released",help="reproduce direct released-value quantization or an explicit F16 intermediate")
    args=parser.parse_args()
    release=verify_release(args.checkpoint)
    records=json.loads((Path(__file__).resolve().parents[2]/"eredu-architectures/tests/fixtures/nemotron_labs_diffusion/conversions.json").read_text())
    record=next(item for item in records if item.get("format")=="gguf" and item["weight_files"][0]["rfilename"]==args.artifact.name)
    artifact=record["weight_files"][0]
    assert release["repo"]==record["base_repository"] and release["revision"]==record["base_revision"]
    assert args.artifact.stat().st_size==artifact["size"]
    assert hashlib.file_digest(args.artifact.open("rb"),"sha256").hexdigest()==artifact["lfs"]["sha256"]
    assert hashlib.sha256((Path(gguf.__file__).resolve().parent/"quants.py").read_bytes()).hexdigest()==record["reference_quants_sha256"]
    configuration=json.loads((args.checkpoint/"config.json").read_text())
    library=ctypes.CDLL(str(args.quantizer_library.resolve()))
    quantize=library.ggml_quantize_chunk
    quantize.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
    quantize.restype=ctypes.c_size_t
    reader=gguf.GGUFReader(str(args.artifact))
    fields={"attn_q":"self_attn.q_proj","attn_k":"self_attn.k_proj","attn_v":"self_attn.v_proj","attn_output":"self_attn.o_proj","attn_norm":"input_layernorm","ffn_norm":"post_attention_layernorm","ffn_gate":"mlp.gate_proj","ffn_up":"mlp.up_proj","ffn_down":"mlp.down_proj"}
    results=[]
    with safe_open(args.checkpoint/"model.safetensors",framework="pt",device="cpu") as source:
        for tensor in reader.tensors:
            name=tensor.name
            if name.startswith("blk."):
                _,index,field,suffix=name.split(".")
                assert field in fields and suffix=="weight"
                source_name=f"encoder.layers.{index}.{fields[field]}.weight"
            else:
                source_name={"token_embd.weight":"encoder.embed_tokens.weight","output_norm.weight":"encoder.norm.weight","output.weight":"diffusion_head.weight"}[name]
            slice=source.get_slice(source_name)
            shape=tuple(slice.get_shape())
            assert shape==tuple(reversed(tensor.shape))
            kind=int(tensor.tensor_type)
            if len(shape)==1:
                values=slice[:].float().numpy()
                expected=values.astype(np.float32 if kind==0 else np.float16).view(np.uint8)
                actual=tensor.data.reshape(-1).view(np.uint8)
                differing=int(np.count_nonzero(expected!=actual))
                results.append({"tensor":name,"type":kind,"bytes":int(actual.size),"differing_bytes":differing})
                continue
            rows,columns=shape
            actual=tensor.data.reshape(rows,-1).view(np.uint8)
            row_bytes=actual.shape[1]
            chunk=max(1,(8<<20)//(columns*4))
            permuted=None
            if record["rotary_pairing"]=="adjacent_pairs" and source_name.endswith(("self_attn.q_proj.weight","self_attn.k_proj.weight")):
                heads=configuration["num_attention_heads"] if source_name.endswith("q_proj.weight") else configuration["num_key_value_heads"]
                permuted=slice[:].float().numpy().reshape(heads,2,configuration["head_dim"]//2,columns).transpose(0,2,1,3).reshape(shape)
            differing=0
            for start in range(0,rows,chunk):
                end=min(start+chunk,rows)
                values=permuted[start:end] if permuted is not None else slice[start:end].float().numpy()
                if args.source_precision=="f16":
                    values=values.astype(np.float16).astype(np.float32)
                values=np.ascontiguousarray(values)
                if kind in (0,1):
                    output=values.astype(np.float32 if kind==0 else np.float16).view(np.uint8)
                else:
                    assert kind in (8,12,13,14)
                    output=np.empty((end-start,row_bytes),dtype=np.uint8)
                    written=quantize(kind,values.ctypes.data,output.ctypes.data,0,end-start,columns,None)
                    assert written==output.size
                differing+=int(np.count_nonzero(output.reshape(end-start,row_bytes)!=actual[start:end]))
            results.append({"tensor":name,"type":kind,"bytes":int(actual.size),"differing_bytes":differing})
            print(f"{name}: differing_bytes={differing}",flush=True)
    result={"repository":record["repo"],"revision":record["revision"],"official_source":record["base_repository"],"official_revision":record["base_revision"],"reference_revision":record["reference_revision"],"source_precision":args.source_precision,"quantizer_library_sha256":hashlib.file_digest(args.quantizer_library.open("rb"),"sha256").hexdigest(),"compared_bytes":sum(item["bytes"] for item in results),"differing_bytes":sum(item["differing_bytes"] for item in results),"tensors":results}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"compared_bytes={result['compared_bytes']} differing_bytes={result['differing_bytes']}",flush=True)


if __name__=="__main__":
    main()
