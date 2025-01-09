import cv2
import os
import json
from collections import Counter

split = 'val'

total_length = 21
single_length = 11

dataset_ids = []
gt_ch1 = os.listdir("/home/mkhan/stitching/data/track2/PanoFeatures/train/")
gt_ch2 = os.listdir("/home/mkhan/stitching/data/track2/PanoFeatures/val/")
#gt_ch3 = os.listdir("/home/mkhan/stitching/data/track2/PanoFeatures/test/")
gt_files1 = [str(img_path).split('.')[0].split('-')[0] for img_path in gt_ch1]
gt_files2 = [str(img_path).split('.')[0].split('-')[0] for img_path in gt_ch2]
counted = Counter(gt_files1)

# Filter to keep only strings that appear exactly twice
filtered_list = [item for item in gt_files1 if counted[item] == 2]

# Remove duplicates while maintaining the order
train = list(dict.fromkeys(filtered_list))

counted = Counter(gt_files2)

# Filter to keep only strings that appear exactly twice
filtered_list = [item for item in gt_files2 if counted[item] == 2]

# Remove duplicates while maintaining the order
val = list(dict.fromkeys(filtered_list))
print(len(val), len(train))  

#gt_files3 = [str(img_path).split('.')[0] for img_path in gt_ch3 if "-aft" in img_path]
''''''
with open('dataset_ids_'+"train"+'.json', 'w') as f:
  json.dump(train, f)
with open('dataset_ids_'+"val"+'.json', 'w') as f:
  json.dump(val, f)
#with open('dataset_ids_'+"test"+'.json', 'w') as f:
#  json.dump(gt_files3, f)