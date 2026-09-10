"""Independent finite top-k partitions with exact ties across the cutoff."""
import hashlib,json
from pathlib import Path
import torch
from safetensors.torch import save_file

torch.set_num_threads(1)
values={}
for width, count in [(64,4),(100,8),(384,8)]:
    # Rational values intentionally produce many equal maxima and cutoff ties.
    choices=((torch.arange(17*width)*117+31)%97).reshape(17,width).float().div(16).bfloat16().float()
    values[f'{width}.input']=choices
    values[f'{width}.indices']=torch.topk(choices,count,dim=-1).indices.int()
path=Path(__file__).with_suffix('.safetensors');save_file(values,path)
path.with_suffix('.json').write_text(json.dumps({'torch':torch.__version__,'torch_git':torch.version.git_version,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'operation':'descending finite top-k with equal cutoff keys'},indent=2)+'\n')
