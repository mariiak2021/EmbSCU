import cv2
import os
import json


split = 'val'

total_length = 15


dataset_ids = []
gt_ch1 = os.listdir("/home/mkhan/stitching/data/track2/Mergedcaptions2/train/")
gt_ch2 = os.listdir("/home/mkhan/stitching/data/track2/Mergedcaptions2/val/")
gt_ch3 = os.listdir("/home/mkhan/stitching/data/track2/Mergedcaptions2/test/")
with open("/home/mkhan/embclip-rearrangement/change_recognition/pano_code/dataset_ids_train.json", 'r') as file:
    datatr = json.load(file)
with open("/home/mkhan/embclip-rearrangement/change_recognition/pano_code/dataset_ids_val.json", 'r') as file:
    dataval = json.load(file)
with open("/home/mkhan/embclip-rearrangement/change_recognition/pano_code/dataset_ids_test.json", 'r') as file:
    datatest = json.load(file)
gt_files1 = ["/home/mkhan/stitching/data/track2/Mergedcaptions2/train/" + str(img_path) for img_path in gt_ch1]
gt_files2 = ["/home/mkhan/stitching/data/track2/Mergedcaptions2/val/" + str(img_path) for img_path in gt_ch2]
gt_files3 = ["/home/mkhan/stitching/data/track2/Mergedcaptions2/test/" + str(img_path) for img_path in gt_ch3]
changes = []
types = []
gt_files = gt_files1 + gt_files2 + gt_files3
with open('ai2thor_changes_word2ids.json', 'r') as f:
  word2ids = json.load(f)
#print (gt_files)
en_changes1 = {}

for filee in gt_files1:
  changes = []
  scene = filee.split('/')[-1].split('.')[0]
  if scene in datatr:
    with open(filee, 'r') as file:
        for line in file:
          elements = line.strip().split(',')
          #print (elements, scene)
          second_element = elements[1].split('_')[0]
          #rint (second_element)
          changes.append (second_element)
  if len(changes) == 1 and changes[0]== " Fridge":
      print(changes, scene)
  for change in changes:
    #print (change)
    if change == "ToiletPaper":
      print (change)
