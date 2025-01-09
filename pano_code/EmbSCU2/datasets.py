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
    print (curr_id)

    img0_0 = Image.open(self.data_folder+'panobef/'+self.split+'/'+curr_id+'-bef.jpg').convert('RGB')
    img0_0 = img0_0.resize((1024, 1024))
    img0_0 = transf(img0_0)
    img0_1 = Image.open(self.data_folder+'panoaft/'+self.split+'/'+curr_id+'-aft.jpg').convert('RGB')
    img0_1 = img0_1.resize((1024, 1024))
    img0_1 = transf(img0_1)


    def compute_centroid(mask):
      """Compute the centroid of a binary mask."""
      indices = np.argwhere(mask > 0)
      if len(indices) == 0:
          return None
      centroid = np.mean(indices, axis=0)
      return centroid

    def absolute_positional_embeddings(masks, panorama_shape):
      """
      Compute absolute positional embeddings for each object mask in a scene.
      
      Args:
          masks (list of np.array): List of binary masks for objects in the scene.
          panorama_shape (tuple): Shape of the panorama image (height, width).
          
      Returns:
          np.array: An array of shape (N, 2) where N is the number of masks,
                    containing the absolute positional embeddings (x, y) for each mask.
      """
      num_masks = len(masks)
      #print (num_masks)
      abs_embeddings = np.zeros((20, 2))
      
      # Compute centroids for all masks
      for i, mask in enumerate(masks):
          centroid = compute_centroid(mask)
          if centroid is not None:
              abs_embeddings[i, 0] = centroid[1] / panorama_shape[1]  # Normalize x-coordinate
              abs_embeddings[i, 1] = centroid[0] / panorama_shape[0]  # Normalize y-coordinate
          else:
              # If no centroid (empty mask), assign None or a special value
              abs_embeddings[i] = [0, 0]
      
      return abs_embeddings

    curr_id = self.dataset_ids[i]

    captions = torch.LongTensor(self.captions[curr_id]).to(device)
    caplens = torch.LongTensor([self.captions_len[curr_id]]).to(device)
    masks = os.listdir("/home/mkhan/stitching/data/track2/OutMasks/" + self.split)
    masks_bef = [np.uint8(cv2.imread("/home/mkhan/stitching/data/track2/OutMasks/" + self.split+'/'+str(img_path), cv2.IMREAD_GRAYSCALE) > 0) for img_path in masks if curr_id in img_path and "bef.png" in img_path]
    masks_aft = [np.uint8(cv2.imread("/home/mkhan/stitching/data/track2/OutMasks/" + self.split+'/'+str(img_path), cv2.IMREAD_GRAYSCALE) > 0) for img_path in masks if curr_id in img_path and "aft.png" in img_path]
    #masks_bef = sorted(masks_bef, key=extract_number_from_filename)
    #print (masks_bef.shape, masks_aft.shape)
    panorama = cv2.imread("/home/mkhan/stitching/data/track2/panobef/" + self.split+'/'+str(curr_id) + "-bef.jpg")
    panorama_shape = panorama.shape[0], panorama.shape[1]
    #print (panorama_shape)
    abs_embeddings_before = absolute_positional_embeddings(masks_bef, panorama_shape)

    # Compute absolute positional embeddings for the 'after' scene
    abs_embeddings_after = absolute_positional_embeddings(masks_aft, panorama_shape)
    #masks_aft = sorted(masks_aft, key=extract_number_from_filename)
    
    img_fea_bef = torch.load(self.data_folder+'PanoFeatures/'+self.split+'/'+curr_id+'-bef.pt', map_location=device)
    img_fea_aft = torch.load(self.data_folder+'PanoFeatures/'+self.split+'/'+curr_id+'-aft.pt',map_location=device)
    
    obj_fea_bef = torch.load(self.data_folder+'PanoObjFeatures/'+self.split+'/'+curr_id+'-bef.pt', map_location=device)
    obj_fea_aft = torch.load(self.data_folder+'PanoObjFeatures/'+self.split+'/'+curr_id+'-aft.pt', map_location=device)
    #print (abs_embeddings_before.shape, abs_embeddings_after.shape, obj_fea_bef.shape, obj_fea_aft.shape)
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

    return img0_0, img0_1, captions, caplens,abs_embeddings_before,abs_embeddings_after,obj_fea_bef,obj_fea_aft,class_bef,class_aft#img0_0, img0_1, captions, caplens,img_fea_bef, img_fea_aft,obj_fea_bef,obj_fea_aft,class_bef,class_aft


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

