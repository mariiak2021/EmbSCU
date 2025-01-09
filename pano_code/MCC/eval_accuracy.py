import os
import numpy as np
import h5py
import json
import torch

from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


with open('/home/mkhan/embclip-rearrangement/change_recognition/pano_code/merged_val2.json', 'r') as f:
  gts = json.load(f)
  

with open('/home/mkhan/embclip-rearrangement/change_recognition/pano_code/MCC/eval_results/39ori_res.json', 'r') as f:
  res = json.load(f)
  
def checktypes(a, b):
  sub_a = []
  sub_b = []
  
  for i in range(0,10,2):
    if len(a) > i:
      sub_a.append(a[i])
      sub_b.append(b[i])
      
  for item in sub_a:
    if item not in sub_b:
      return False
    
  for item in sub_b:
    if item not in sub_a:
      return False    

  return True      

change = 0
unchange = 0
cchange_number = 0
cchange_type = 0
cchange = 0
cunchange = 0
cchange_all = 0
cchange_obj = 0
  
for i in range(434):#3896
  #print (gts[i][0])
  if len(gts[i][0]) == 0:
    unchange = unchange + 1
    
    if len(res[str(i)][0]) == 0:
      cunchange = cunchange + 1
  
  if len(gts[i][0]) != 0:
    change = change + 1
    #print (res[str(i)][0])
    cchange_all = cchange_all + len(gts[i][0].split(' '))/2
    #print (res[str(i)][0].split(' '), gts[i][0].split(' '))
    if len(res[str(i)][0].split(' ')) ==   len(gts[i][0].split(' ')):
      
      cchange_number = cchange_number + 1
    
    #if checktypes(res[str(i)][0].split(' '), gts[i][0].split(' ')):
    #  cchange_type = cchange_type + 1
    #print ("Pr:",res[str(i)][0],"GT:",gts[i][0])
    if res[str(i)][0] == gts[i][0]:
      cchange = cchange + 1

    temp_b = res[str(i)][0].split(' ')#[:-1]
    #print (temp_b)
    temp_a = gts[i][0].split(' ')#[:-1]  
    #print (temp_a)
    for it in range(len(temp_a)):
      if it%2 == 1:
        
        #print ("yes")
        continue
      #print ("no")
      flag = 0
      for jt in range(len(temp_b)):
        if jt%2 == 1:
          continue
        
        if temp_a[it] == temp_b[jt]:
          #print (temp_a[it])
          cchange_type = cchange_type + 1
          break

    for it in range(len(temp_a)):
      if it%2 == 0:
        continue
      
      flag = 0
      for jt in range(len(temp_b)):
        if jt%2 == 0:
          continue
        
        if temp_a[it] == temp_b[jt]:
          #print (temp_a[it])
          cchange_obj = cchange_obj + 1
          break  

print('total: ', float(cchange+cunchange)/(change+unchange))
#print('unchange: ', float(cunchange)/(unchange))
print('change: ', float(cchange)/(change))
print('change, number: ', float(cchange_number)/(change))
print('change, type: ', float(cchange_type)/(cchange_all))
print('change, obj: ', float(cchange_obj)/(cchange_all))

