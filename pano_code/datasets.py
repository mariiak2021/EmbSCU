import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
from torchvision import models, transforms

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


    #img0_0 = Image.open(self.data_folder+'panobef/'+self.split+'/'+curr_id+'-bef.jpg').convert('RGB')
    #img0_0 = transf(img0_0)
    #img0_1 = Image.open(self.data_folder+'panoaft/'+self.split+'/'+curr_id+'-aft.jpg').convert('RGB')
    #img0_1 = transf(img0_1)




    curr_id = self.dataset_ids[i]

    captions = torch.LongTensor(self.captions[curr_id]).to(device)
    caplens = torch.LongTensor([self.captions_len[curr_id]]).to(device)
    img_fea_bef = torch.load(self.data_folder+'PanoFeatures/'+self.split+'/'+curr_id+'-bef.pt', map_location=device)
    img_fea_aft = torch.load(self.data_folder+'PanoFeatures/'+self.split+'/'+curr_id+'-aft.pt',map_location=device)
    obj_fea_bef = torch.load(self.data_folder+'PanoObjFeatures/'+self.split+'/'+curr_id+'-bef.pt', map_location=device)
    obj_fea_aft = torch.load(self.data_folder+'PanoObjFeatures/'+self.split+'/'+curr_id+'-aft.pt', map_location=device)
    # Load class labels from text files
    with open(self.data_folder + 'PanoObjClasses/' + self.split + '/' + curr_id + '-bef.txt', 'r') as file:
            class_bef = file.read().strip().split('\n')  # Assuming classes are newline-separated
    with open(self.data_folder + 'PanoObjClasses/' + self.split + '/' + curr_id + '-aft.txt', 'r') as file:
            class_aft = file.read().strip().split('\n')

        # Convert class labels to LongTensor (assuming they are integer class labels)
    class_bef = [int(entity_classes.index(c)) for c in class_bef]
    class_bef = torch.tensor(class_bef).to(device)
    class_aft = [int(entity_classes.index(c)) for c in class_aft]
    class_aft = torch.tensor(class_aft).to(device)

    return captions, caplens,img_fea_bef, img_fea_aft,obj_fea_bef,obj_fea_aft,class_bef,class_aft#img0_0, img0_1, captions, caplens,img_fea_bef, img_fea_aft,obj_fea_bef,obj_fea_aft,class_bef,class_aft


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

