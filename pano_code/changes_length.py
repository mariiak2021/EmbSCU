import cv2
import os
import json


split = 'val'

total_length = 21
single_length = 11

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
#print (gt_files)
ln_changes1 = {}
for filee in gt_files1:
  scene = filee.split('/')[-1].split('.')[0]
  if scene in datatr:
    
    with open(filee, 'r') as file:
      k = 0
      for line in file:
        k = k + 1
        print (scene, line)
      line_count = sum(1 for line in file)
    print (scene, k, k*2 + 2)
    ln_changes1[scene] = k*2 + 2
  

with open('encoded_changes_length_'+"train"+'.json', 'w') as f:
  json.dump(ln_changes1, f)

ln_changes2 = {}
for filee in gt_files2:
  scene = filee.split('/')[-1].split('.')[0]
  if scene in dataval:
    
    with open(filee, 'r') as file:
      k = 0
      for line in file:
        k = k + 1
        print (scene, line)
      line_count = sum(1 for line in file)
    print (scene, k, k*2 + 2)
    ln_changes2[scene] = k*2 + 2

with open('encoded_changes_length_'+"val"+'.json', 'w') as f:
  json.dump(ln_changes2, f)

ln_changes3 = {}
for filee in gt_files3:
  scene = filee.split('/')[-1].split('.')[0]
  if scene in datatest:
    
    with open(filee, 'r') as file:
      k = 0
      for line in file:
        k = k + 1
        print (scene, line)
      line_count = sum(1 for line in file)
    print (scene, k, k*2 + 2)
    ln_changes3[scene] = k*2 + 2

with open('encoded_changes_length_'+"test"+'.json', 'w') as f:
  json.dump(ln_changes3, f)

  

