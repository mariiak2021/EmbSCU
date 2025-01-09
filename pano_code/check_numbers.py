import cv2
import os
import json


with open('encoded_changes_length_train.json', 'r') as f:
  curr = json.load(f)
  
max_c = 0
for key in curr:
  if curr[key] > max_c:
    max_c = curr[key]
    
print(max_c)
