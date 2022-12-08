# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb

from dataset import from_path as dataset_from_path
from model import DiffuSE
from params import AttrDict


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffuSELearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta) 
    # cumulative product of alpha, which is 1 - beta
    # noise_level = alpha bar, which is used to specify a specific distribution from the original image
    # essentially letting us skip over the iterative process.
    # explanation, kind of: https://youtu.be/a4Yfz2FxXiY?t=651

    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict, pretrain=False):
    if pretrain:
      print("WARNING: Remove parameters from model")
      model_state_dict = state_dict['model']
      for i in range(30):
        model_state_dict.pop("residual_layers.{}.conditioner_projection.weight".format(i), None)
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      if pretrain:
        # Missing key(s) in state_dict:"residual_layers.{i}.conditioner_projection.weight" 
        # Missing key(s) in state_dict:"residual_layers.{i}.output_residual.{weight,bias}" 
        self.model.module.load_state_dict(model_state_dict, strict=False)
      else:
        self.model.module.load_state_dict(state_dict['model'])
    else:
      if pretrain:
        self.model.load_state_dict(model_state_dict, strict=False)
      else:
        self.model.load_state_dict(state_dict['model'])
    
    if not pretrain:
      self.optimizer.load_state_dict(state_dict['optimizer'])
      self.scaler.load_state_dict(state_dict['scaler'])
      self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self,pretrain_path=None, filename='weights'):
    # pdb.set_trace()
    if pretrain_path!=None:
      print(f'load pretrain model at {pretrain_path}')
      checkpoint = torch.load(pretrain_path)
      self.load_state_dict(checkpoint,pretrain=True)
    else:
      try:
        checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
        self.load_state_dict(checkpoint)
        return True
      except FileNotFoundError:
        return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    noisy = features['noisy']
    spectrogram = features['spectrogram']

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device) 
      # get the t step from the noise schedule (the one that interpolates from clean to full noisy)
      # question is, should what kind of noise_schedule should we use?
      # CDiffuSE uses a linear noise schedule, (see: cDiffuSE, 4.1, Model Architecture and Training)
      # cold diffuse uses a cosine noise schedule, (see: CD, 4.2, Model Setting and Training Procedure)
      # to change the noise schedule, change it in the params in params.py
      # noise_schedule=np.linspace(1e-4, 0.035, 50).tolist(), (this is base CDiffuSE)

      noise_scale = self.noise_level[t].unsqueeze(1) # gets the noise_level at the specific step we want, which is defined by t.
      noise_scale_sqrt = noise_scale**0.5 # square root of noise_scale lol

      m = (((1-self.noise_level[t])/self.noise_level[t]**0.5)**0.5).unsqueeze(1) 
      # / (1-a_bar)  \ 0.5
      # | —————————  |        = m_t  (interpolation coefficient at step t)
      # \ a_bar**0.5 /

      noise = torch.randn_like(audio) # generates Gaussian noise (1d)

      noisy_audio = (1-m) * noise_scale_sqrt  * audio + m * noise_scale_sqrt * noisy  + (1.0 - (1+m**2) *noise_scale)**0.5 * noise
      # equation ?? 
      # interpolate between clean audio, noisy audio, and adding gaussian noise.
      # qcdiff (xt | x0, y) + gaussian noise (eq. 8)


      combine_noise = (m * noise_scale_sqrt * (noisy-audio) + (1.0 - (1+m**2) *noise_scale)**0.5 * noise) / (1-noise_scale)**0.5
      # (noisy-audio) = scaled real, background noise.
      # (1.0 - (1+m**2) *noise_scale)**0.5 * noise) = scaled gaussian noise. scale by variance term of noisy audio
      # (1.0 - (1+m**2) *noise_scale)**0.5 = variance term of the noisy audio.
      # so this is what we use in the loss function.
      # combined noise, is literally the real bg noise + the gaussian noise.

      predicted = self.model(noisy_audio, spectrogram, t)
      # okay... to match the cold diffusion paper, we'll somehow have to further degrade the prediction
      # and then predict from the redegraded prediction, and then change the loss function to test both of these things

      # maybe copy paste the degradation code and run it again? but this time on the predicted audio?
      # we will have to reuse the noisy audio, and the noisy spectrogram.
      # then create a second prediction variable, and then pass to loss?

      # first, we'll have to replicate the inputs to the prediction.
      # noisy_audio = interpolated clean/noisy + gaussian noise
      # spectrogram = spectrogram of the noise
      # t = step. which should be LESS than the previous t

      ### NEW CODE FROM HERE ON OUT --------------


      # loss is calculated on
      # loss = predicted - (gaussian + noise)

      # redefine audio = noisy + gaussian - predicted (
      #       predicted = gaussian + noise, 
      #       noisy = clean + noise
      #       clean = noisy + gaussian - predicted 
      #       clean = (clean + noise) + gaussian - (noise - gaussian))
      #       gaussians are supposed to be the same.
      # is getting the clean audio as simple as that?
      debug = False  
    
    
      if debug:
          print("First noisy_audio shape :", noisy_audio.shape)
          print("clean" , audio.shape)
          print("combined", combine_noise.shape)
          print("predicted", predicted[:,0,:].shape)
          print("noise_scale", noise_scale.shape)
          print("noise", noise.shape)
          print("m", m.shape)
    

      predicted_clean = audio + combine_noise - predicted[:,0,:] # for some reason, this has 3 dimensions, with 1 dimension have only 1 layer
      # predicted_clean = clean + (noise + gaussian) - (p_noise + p_gaussian)

      temp_t = torch.Tensor().long()
      for i in t: # loop through t in the previous degradation, and randomly pick a new t for each sample in the batch.
          random_max = int(i) if int(i) > 0 else 1 # to avoid max of randint being 0, which throws an error
          temp_t = torch.cat((temp_t, torch.randint(0, random_max, (1,)).unsqueeze(0)), 1)
          # t must be less than the previous t.
      t = temp_t[0,:]
        
      
      #t = torch.randint(0, t.median().item(), [N], device=audio.device) 
      # t must be less than the previous t.
      # second degradation must be less than the first degradation

      noise_scale = self.noise_level[t].unsqueeze(1) # gets the noise_level at the specific step we want, which is defined by t.
      noise_scale_sqrt = noise_scale**0.5 # square root of noise_scale lol
      m = (((1-self.noise_level[t])/self.noise_level[t]**0.5)**0.5).unsqueeze(1) 
      noise = torch.randn_like(predicted_clean) # comment out if using same noise for both degradations
      noisy_audio = (1-m) * noise_scale_sqrt  * predicted_clean + m * noise_scale_sqrt * noisy  + (1.0 - (1+m**2) *noise_scale)**0.5 * noise
        
      combine_noise_2nd = (m * noise_scale_sqrt * (noisy-audio) + (1.0 - (1+m**2) *noise_scale)**0.5 * noise) / (1-noise_scale)**0.5

      if debug:
          print("Second noisy_audio shape :", noisy_audio.shape)
          print("clean" , predicted_clean.shape)
          print("noise_scale", noise_scale.shape)
          print("noise", noise.shape)
          print("m", m.shape)
      re_predicted =  self.model(noisy_audio, spectrogram, t)

      ### NEW CODE ENDS HERE --------------

      # loss would look something like
      # loss_fn is actually just an L1 loss function.
      # loss = self.loss_fn(combine_noise, predicted.squeeze(1)) + self.loss_fn(combine_noise, repredicted.squeeze(1))
      # first, we have to figure out what combine_noise actually is.

      # so, turns out... the loss function is not comparing the clean audio against the predicted audio!
      # instead, it's actually comparing the combined noise, what we predict the noise is!
      # so, noisy audio + gaussian noise - predicted combined_nose = predicted clean noise!
      # how do we use the output as another input?
      loss = (self.loss_fn(combine_noise, predicted.squeeze(1)) + self.loss_fn(combine_noise_2nd, re_predicted.squeeze(1)))/2

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffuSELearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint(args.pretrain_path)
  learner.train(max_steps=args.max_steps)


def train(args, params):
  dataset = dataset_from_path(args.clean_dir, args.noisy_dir, args.data_dirs, params, se=args.se, voicebank=args.voicebank)
  model = DiffuSE(args, params).cuda()
  _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffuSE(args, params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=True)
  _train_impl(replica_id, model, dataset_from_path(args.clean_dir, args.noisy_dir, args.data_dirs, params, se=args.se, voicebank=args.voicebank, is_distributed=True), args, params)
