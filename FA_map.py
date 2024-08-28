# -*- coding: utf-8 -*-
"""

@author: Shaheim Ogbomo-Harmitt

Class to evaluate thresholded feature atttribution maps(LIME, GradCAM and Occlusions)

FA(FA_type,model_pth,image_pth,strat_index):
    
    Inputs:
        
        FA_type (str) - feature attribustion method. GradCAM = 'GradCAM', LIME = 'LIME' and Occlusions = 'OCC'.
        
        model_pth (str) - Pytorch file (.pt) of saved model state.
        
        image_pth (str) - file path of raw image (.jpg file) to apply feature attribution map to.
        
        strat_index (int) - index of ablation strategy from DL model output
        
    
    Outputs:
        
        heatmap (array) - thresholded feature attribution map 
    
"""

from lime import lime_image
import numpy as np
from captum.attr import Occlusion
from PIL import Image
import cv2 
from skimage import io,color
from Network import *
import torch
from captum.attr import LayerGradCam

class FA:
    
  def __init__(self,FA_type,model_pth,image_pth,strat_index):
      
    self.FA_type = FA_type
    self.model_pth = model_pth
    self.image_pth = image_pth
    self.strat_index = strat_index
        
  def Get_Data(self):

    image = Image.open(self.image_pth)
    data = np.asarray(image)
    data = cv2.resize(data, (150,150), interpolation=cv2.INTER_LINEAR)
        
    return data
    
  def normalise(self,a):
        
    b = (a - np.min(a))/np.ptp(a)
        
    return b
    
  def preprocess(self,image):
        
    image = color.rgb2gray(image)
    image = image - image.mean()
    image = image/image.max()
    image = image[None,:,:]
        
    return torch.tensor(image).float()
    
  def Occlusion_func(self,input_image,feature_index,model):
    
    image = color.rgb2gray(input_image)
    disk_mask = np.round(image,1) != 0.7
    input_image = self.preprocess(input_image)
        
    input_image = input_image.unsqueeze(1)
        
    ablator = Occlusion(model)
        
    attr = ablator.attribute(input_image, target = feature_index, sliding_window_shapes=(1,25,25))
        
    attributions = attr.detach().cpu().numpy()
        
    Attribution = attributions[0,0,:,:]
        
    Attribution = self.normalise(Attribution)
    Attribution = Attribution * disk_mask
    
    return Attribution


  def Lime_Image_Predict(self,images):
    
    batch = torch.stack(tuple(self.preprocess(i) for i in images))
    model = Net()
    model.load_state_dict(torch.load(self.model_pth))
    model.eval()
    outputs = model(batch)
    
    return outputs.detach().cpu().numpy()

  def LIME_func(self,input_image,strat_index):
    
    # Get Data 
    
    image = color.rgb2gray(input_image)
    disk_mask = np.round(image,1) != 0.7
    input_image = self.preprocess(input_image)
    
    results = self.Lime_Image_Predict(input_image)
    
    sort_indices = np.argsort(results)[::-1][0]
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image,self.Lime_Image_Predict,top_labels=3,num_samples= 50) 
    dict_heatmap = dict(explanation.local_exp[sort_indices[strat_index]])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    heatmap = self.normalise(heatmap)
    new_heatmap = heatmap * disk_mask
    
    return new_heatmap

  def GRAD_CAM_func(self,input_image,feature_index,model):
    
    image = color.rgb2gray(input_image)
    disk_mask = np.round(image,1) != 0.7
    input_image = self.preprocess(input_image)
    
    input_image = input_image.unsqueeze(1)
    
    guided_gc = LayerGradCam(model, model.conv4)
    
    attributions = guided_gc.attribute(input_image, feature_index,relu_attributions = True)
    
    attributions = attributions.detach().cpu().numpy()
    
    Attribution = attributions[0,0,:,:]
    Attribution = cv2.resize(Attribution, (150,150), interpolation=cv2.INTER_LINEAR)
    Attribution = Attribution
    Attribution = self.normalise(Attribution)
    Attribution = Attribution * disk_mask
    
    return Attribution
    
  def run(self):
    
    data = self.Get_Data()
    model = Net()
    model.load_state_dict(torch.load(self.model_pth))
    model.eval()
    
    if self.FA_type == 'OCC':
        
        heatmap = self.Occlusion_func(data,self.strat_index,model)
        heatmap = heatmap > np.mean(heatmap)
        
    elif self.FA_type == 'GradCAM':
        
        heatmap = self.GRAD_CAM_func(data,self.strat_index,model)
        heatmap = heatmap > np.mean(heatmap)
        
    elif self.FA_type == 'LIME':
        
        heatmap = self.LIME_func(data,self.strat_index)
        heatmap = heatmap > np.mean(heatmap)
        
    return heatmap
    
