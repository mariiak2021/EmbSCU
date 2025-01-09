import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
from torchvision import models, transforms
import cv2, numpy as np

device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transf = transforms.ToTensor()

class CaptionDataset(Dataset):
  def __init__(self, data_folder, data_name, split):
    self.split = split
    assert self.split in {'train', 'val', 'test'}
    
    self.data_folder = data_folder

    with open(os.path.join("/home/mkhan/embclip-rearrangement/change_recognition/pano_code/", 'dataset_ids_'+self.split+'.json'), 'r') as f: ##TODO##
      self.dataset_ids = json.load(f)

    with open(os.path.join("/home/mkhan/embclip-rearrangement/change_recognition/pano_code/", 'encoded_changes_'+self.split+'.json'), 'r') as f: ##TODO##
      self.captions = json.load(f)              
      
    with open(os.path.join("/home/mkhan/embclip-rearrangement/change_recognition/pano_code/", 'encoded_changes_length_'+self.split+'.json'), 'r') as f: ##TODO##
      self.captions_len = json.load(f)    

    self.dataset_size = len(self.captions)

  def __getitem__(self, i):
    #print (self.dataset_ids)
    #print (i)
    curr_id = self.dataset_ids[i]#.replace('change_description','')
    #print (curr_id)

    img0_0 = Image.open(self.data_folder+'panobef/'+self.split+'/'+curr_id+'-bef.jpg').convert('RGB')
    img0_0 = img0_0.resize((1200, 300))
    img0_0 = transf(img0_0)
    img0_1 = Image.open(self.data_folder+'panoaft/'+self.split+'/'+curr_id+'-aft.jpg').convert('RGB')
    img0_1 = img0_1.resize((1200, 300))
    img0_1 = transf(img0_1)

    
    curr_id = self.dataset_ids[i]

    captions = torch.LongTensor(self.captions[curr_id]).to(device)
    caplens = torch.LongTensor([self.captions_len[curr_id]]).to(device)
    

    return img0_0, img0_1, captions, caplens


  def __len__(self):
    return self.dataset_size

  
  
entity_classes = [
        "AlarmClock",
        "ArmChair",
        "BaseballBat",
        "BasketBall",
        "Bathtub",
        "Bed",
        "Blinds",
        "Book",
        "Bowl",
        "Box",
        "Cabinet",
        "CoffeeTable",
        "CounterTop",
        "Desk",
        "DiningTable",
        "Drawer",
        "Dresser",
        "Fridge",
        "GarbageCan",
        "Laptop",
        "LaundryHamper",
        "Microwave",
        "Mug",
        "Newspaper",
        "Ottoman",
        "Pan",
        "PaperTowelRoll",
        "Plate",
        "Plunger",
        "Pot",
        "Safe",
        "Shelf",
        "ShowerCurtain",
        "ShowerDoor",
        "SideTable",
        "Sink",
        "SoapBottle",
        "Sofa",
        "Statue",
        "StoveBurner",
        "TVStand",
        "TissueBox",
        "Toilet",
        "ToiletPaper",
        "ToiletPaperHanger",
        "Vase",
        "WateringCan",
        "ScrubBrush",
        "None"
]

