"""Independent masked Float32 softmax with BF16 output rounding."""
from pathlib import Path
import torch,json,hashlib
from safetensors.torch import save_file
torch.set_num_threads(1)
fixture={}
for width in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,64,181,193,207,513]:
 x=(torch.sin(torch.arange(64*width,dtype=torch.float32)*.37+.1)*7.5).reshape(64,width).bfloat16().float()
 x=x.masked_fill(torch.arange(width)[None,:] > (torch.arange(64)[:,None]*3)%width,float('-inf'))
 fixture[f'{width}.input']=x;fixture[f'{width}.output']=torch.softmax(x,dim=-1).bfloat16()
p=Path(__file__).with_suffix('.safetensors');save_file(fixture,p)
p.with_suffix('.json').write_text(json.dumps(dict(torch=torch.__version__,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2)+'\n')
