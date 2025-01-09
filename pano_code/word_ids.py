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
gt_files1 = ["/home/mkhan/stitching/data/track2/Mergedcaptions2/train/" + str(img_path) for img_path in gt_ch1]
gt_files2 = ["/home/mkhan/stitching/data/track2/Mergedcaptions2/val/" + str(img_path) for img_path in gt_ch2]
gt_files3 = ["/home/mkhan/stitching/data/track2/Mergedcaptions2/test/" + str(img_path) for img_path in gt_ch3]
changes = []
types = []
gt_files = gt_files1 + gt_files2 + gt_files3
print (gt_files)
for filee in gt_files:
  with open(filee, 'r') as file:
    for line in file:
      elements = line.strip().split(',')
      #print (elements)
      second_element = elements[1].split('_')[0]
      first_element = elements[0].strip()
      changes.append(second_element)
      types.append(first_element)
print (types)
print (changes)
all = types + changes 
word2ids = {}
word2ids['<pad>'] = 0
word2ids['<start>'] = 1
word2ids['<end>'] = 2

  
for item in all:
    #print (item, word2ids)
    if item not in word2ids:
      word2ids[item] = len(word2ids)



  
with open('ai2thor_changes_word2ids.json', 'w') as f:
  json.dump(word2ids, f)
  
