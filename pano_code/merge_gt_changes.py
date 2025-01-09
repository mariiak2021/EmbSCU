import cv2
import os
import json





gt_ch2 = os.listdir("/home/mkhan/stitching/data/track2/Mergedcaptions2/val/")


with open("/home/mkhan/embclip-rearrangement/change_recognition/pano_code/dataset_ids_val.json", 'r') as file:
    dataval = json.load(file)

gt_files2 = ["/home/mkhan/stitching/data/track2/Mergedcaptions2/val/" + str(img_path) for img_path in gt_ch2]


en_changes2 = []
'''
for filee in gt_files2:
  changes = []
  scene = filee.split('/')[-1].split('.')[0]
  if scene in dataval:
    with open(filee, 'r') as file:
        for line in file:
          elements = line.strip().split(',')
          #print (elements)
          second_element = elements[1].split('_')[0][1:]
          first_element = elements[0].strip()
          changes.append(first_element)
          changes.append(second_element)
    combined_string = combined_string = " ".join(words)#" ".join(word.strip() for word in changes)
        
    en_changes2.append([combined_string])'''


for scene in dataval:
    changes = []
    filee = "/home/mkhan/stitching/data/track2/Mergedcaptions2/val/" + scene + ".txt"
    with open(filee, 'r') as file:
        for line in file:
          elements = line.strip().split(',')
          #print (elements)
          second_element = elements[1].split('_')[0][1:]
          first_element = elements[0].strip()
          changes.append(first_element)
          changes.append(second_element)
    combined_string = combined_string = " ".join(word.strip() for word in changes)
        
    en_changes2.append([combined_string])

with open('merged_val2.json', 'w') as f:
  json.dump(en_changes2, f)
