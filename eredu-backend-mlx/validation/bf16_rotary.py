"""Independent input-product YaRN rounding over short and distant positions."""
from pathlib import Path
import math,json,hashlib
import torch
from safetensors.torch import save_file
torch.set_num_threads(1)
base=1_000_000.;width=64;factor=16.;context=8192;fast=128.;slow=4.;amplitude=1.2772588722239782
correction=lambda n: width*math.log(context/(n*2*math.pi))/(2*math.log(base))
low=max(math.floor(correction(fast)),0);high=min(math.ceil(correction(slow)),width-1)
pos_freqs=base**(torch.arange(0,width,2,dtype=torch.float32)/width)
ramp=((torch.arange(width//2,dtype=torch.float32)-low)/(high-low)).clamp(0,1)
mask=1-ramp
inv=(1/(factor*pos_freqs))*(1-mask)+(1/pos_freqs)*mask
fixture={}
for offset in [0,128,32768]:
 x=(torch.sin(torch.arange(2*193*width,dtype=torch.float32)*.37+.1)*2.5).reshape(1,2,193,width).bfloat16()
 angles=torch.arange(offset,offset+193,dtype=torch.float32)[:,None]*inv
 cos=(angles.cos()*amplitude).bfloat16()[None,None];sin=(angles.sin()*amplitude).bfloat16()[None,None]
 first,last=x[...,:width//2],x[...,width//2:]
 y=torch.cat((first*cos-last*sin,last*cos+first*sin),-1)
 fixture[f'{offset}.input']=x;fixture[f'{offset}.output']=y
default_width=128
default_inv=1/(10_000_000.**(torch.arange(0,default_width,2,dtype=torch.float32)/default_width))
for offset in [0,128,32768]:
 x=(torch.sin(torch.arange(2*193*default_width,dtype=torch.float32)*.37+.1)*2.5).reshape(1,2,193,default_width).bfloat16()
 angles=torch.arange(offset,offset+193,dtype=torch.float32)[:,None]*default_inv
 cos=angles.cos().bfloat16()[None,None];sin=angles.sin().bfloat16()[None,None]
 first,last=x[...,:default_width//2],x[...,default_width//2:]
 fixture[f'default.{offset}.input']=x
 fixture[f'default.{offset}.output']=torch.cat((first*cos-last*sin,last*cos+first*sin),-1)
path=Path(__file__).with_suffix('.safetensors');save_file(fixture,path)
Path(__file__).with_suffix('.json').write_text(json.dumps(dict(torch=torch.__version__,width=width,base=base,factor=factor,original_context=context,beta_fast=fast,beta_slow=slow,amplitude=amplitude,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
