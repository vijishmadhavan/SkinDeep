import gradio as gr
import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np4
import urllib.request
import PIL.Image
from io import BytesIO
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np
import urllib.request
from urllib.request import urlretrieve
import PIL.Image
from io import BytesIO
import torchvision.transforms as T
import torchvision.transforms as tfms

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]
 
    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

MODEL_URL = "https://www.dropbox.com/s/vxgw0s7ktpla4dk/SkinDeep2.pkl?dl=1"
urllib.request.urlretrieve(MODEL_URL, "SkinDeep2.pkl")
path = Path(".")
learn=load_learner(path, 'SkinDeep2.pkl')

urlretrieve("https://pyxis.nymag.com/v1/imgs/ed6/c4f/367aedb908ac63e80936c174423cc4ac57-26-adam-levine-tattoos-005.2x.h473.w710.jpg","soccer1.jpg")
urlretrieve("https://i.insider.com/5e56b2e8a9f40c0442550f4a?auto=webp&enable=upscale&fit=crop&quality=85&width=832&height=624","soccer2.jpg")
urlretrieve("https://i.insider.com/5e56bfd3a9f40c1797218489?auto=webp&enable=upscale&fit=crop&quality=85&width=832&height=624","baseball.jpg")
urlretrieve("https://wallpapercave.com/dwp1x/wp1990689.jpg","baseball2.jpeg")

sample_images = [["soccer1.jpg"],
                 ["soccer2.jpg"],
                 ["baseball.jpg"],
                 ["baseball2.jpeg"]]


def predict(input):
  img_t = T.ToTensor()(input)
  img_fast = Image(img_t)
  p,img_hr,b = learn.predict(img_fast)
  x = np.minimum(np.maximum(image2np(img_hr.data*255), 0), 255).astype(np.uint8)
  img = PIL.Image.fromarray(x)
  return img 

gr_interface = gr.Interface(fn=predict, inputs=gr.inputs.Image(), outputs="image", title='SkinDeep',examples=sample_images).launch();
