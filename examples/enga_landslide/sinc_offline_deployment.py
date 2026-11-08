"""Single-file offline deployment bundle for the two-stage baseline (not neural distillation)."""
import io,json,tempfile,zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from onboard.core.onboard_inference import OnboardInference
from ground.core.NDCUtils import normalize_time

class Deployment:
 def __init__(self,path,device='cuda'):
  self.temp=tempfile.TemporaryDirectory(prefix='sinc_deploy_')
  with zipfile.ZipFile(path) as z:
   self.manifest=json.loads(z.read('manifest.json'));p=Path(self.temp.name)/'prior.gndc';p.write_bytes(z.read('prior.gndc'))
   with np.load(io.BytesIO(z.read('state.npz'))) as a:ids=a['pixel_ids'].copy();co=a['coefficients'].copy();sc=a['residual_scale'].copy()
  self.inf=OnboardInference(str(p),device=device);self.meta=self.inf.meta;self.device=device;self.H=self.meta['height'];self.W=self.meta['width']
  self.ids=ids;self.lookup=torch.full((self.H*self.W,),-1,dtype=torch.long,device=device);self.lookup[torch.as_tensor(ids,device=device)]=torch.arange(len(ids),device=device)
  self.coefficients=torch.tensor(co,device=device);self.scale=torch.tensor(sc,device=device)
  self.coords=self.inf._create_coords_grid(self.H,self.W)
  self.start=datetime.fromisoformat(self.meta['global_start_date']);self.end=datetime.fromisoformat(self.meta['global_end_date'])
 def query(self,date,pixel_ids=None):
  pixels=torch.arange(self.H*self.W,device=self.device) if pixel_ids is None else torch.as_tensor(pixel_ids,dtype=torch.long,device=self.device)
  index=self.lookup[pixels]
  if (index<0).any():raise ValueError('Deployment file does not cover requested pixels')
  outputs=[];sigmas=[];tn=normalize_time(date,self.start,self.end);day=(date-self.start).total_seconds()/86400
  phi=torch.tensor([1.]+[f(2*np.pi*k*day/365.25) for k in range(1,4) for f in [np.cos,np.sin]],device=self.device,dtype=torch.float32)
  with torch.no_grad():
   for first in range(0,len(pixels),262144):
    pos=pixels[first:first+262144];ind=index[first:first+262144];t=torch.full((len(pos),1),tn,device=self.device)
    mu=self.inf.model.model.forward_with_state(self.coords[pos],t)[0].float();lo,hi=self.meta.get('data_range',[0,1]);mu=mu*(hi-lo)+lo
    if self.meta.get('use_log',False):mu=torch.expm1(mu)
    mu=mu+torch.einsum('k,nkb->nb',phi,self.coefficients[ind]);outputs.append(mu.cpu().numpy());sigmas.append(self.scale[ind].cpu().numpy())
  return np.concatenate(outputs),np.concatenate(sigmas)
 def close(self):self.temp.cleanup()

def export_bundle(model,correction,scale,output,case,smoke=False):
 from run_all_adaptive_comparison import sha,read_header
 m=read_header(model)['dataset_meta'];info=json.loads(scale.with_suffix('.json').read_text())
 if info['model_sha256']!=sha(model) or info['correction_sha256']!=sha(correction):raise ValueError('Inconsistent mean and scale states')
 with np.load(correction) as z:ids=z['pixel_ids'].copy();coeff=z['coefficients'].copy()
 with np.load(scale) as z:
  if not np.array_equal(ids,z['pixel_ids']):raise ValueError('State/scale coverage differs')
  sigma=z['scale'].copy()
 if not smoke and len(ids)!=m['height']*m['width']:raise ValueError('Full-scene export requires full adaptive state coverage')
 state=io.BytesIO();np.savez_compressed(state,pixel_ids=ids,coefficients=coeff.astype('float32'),residual_scale=sigma.astype('float32'))
 meta=dict(case=case,height=m['height'],width=m['width'],covered_pixels=len(ids),smoke=smoke,model_sha256=sha(model),correction_sha256=sha(correction),scale_sha256=sha(scale),storage='Neural prior plus losslessly compressed float32 per-pixel state adjustments and scales; not a fully neural parameter representation',coefficient_float32_max_error=float(abs(coeff-coeff.astype('float32')).max()))
 output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as z:z.write(model,'prior.gndc');z.writestr('state.npz',state.getvalue());z.writestr('manifest.json',json.dumps(meta,indent=2))
 print('Exported:',output,'bytes',output.stat().st_size,flush=True)
