"""Independent nonzero eager BF16 attention across bounded query tile sizes."""
from pathlib import Path
import torch,json,hashlib
from safetensors.torch import save_file
torch.set_num_threads(1)
fixture={}
for width in [6,7,8,9,10,93,193,513]:
 def values(heads,phase):
  return (torch.sin(torch.arange(heads*width*64,dtype=torch.float32)*.037+phase)*1.5).reshape(1,heads,width,64).bfloat16()
 q,k,v=values(4,.1),values(2,.7),values(2,1.2)
 mask=torch.arange(width)[None,:]<=torch.arange(width)[:,None]
 scores=((q@k.repeat_interleave(2,1).transpose(-1,-2))*.125).masked_fill(~mask,float('-inf'))
 probabilities=torch.softmax(scores.float(),dim=-1).bfloat16()
 output=probabilities@v.repeat_interleave(2,1)
 for name,tensor in [('queries',q),('keys',k),('values',v),('output',output),('scores',scores),('probabilities',probabilities)]: fixture[f'{width}.{name}']=tensor
p=Path(__file__).with_suffix('.safetensors');save_file(fixture,p)
p.with_suffix('.json').write_text(json.dumps(dict(torch=torch.__version__,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2)+'\n')
