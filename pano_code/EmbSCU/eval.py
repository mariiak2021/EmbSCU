import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import argparse
import torchvision.models as models
# Parameters
from torch import nn


data_name = 'ai2thor_changes' # base name shared by data files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # sets device for model and PyTorch tensors
cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

batch_size = 1

# model_name

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def evaluate(args, beam_size):
  # Load model
  checkpoint = torch.load(args.checkpoint,map_location='cuda:0')

  encoder = checkpoint['encoder']
  encoder = encoder.to(device)
  encoder.eval()

  decoder = checkpoint['decoder']
  decoder = decoder.to(device)
  decoder.eval()

  '''backbone1 = models.resnet101(pretrained=True)
  features1 = list(backbone1.children())[:-2]
  res101 = nn.Sequential(*features1).cuda()  
  res101.eval()'''

  # Load word map (word2ix)
  with open(args.word_map_file, 'r') as f:
    word_map = json.load(f)

  rev_word_map = {v: k for k, v in word_map.items()}
  vocab_size = len(word_map)

  result_json_file = {}

  # DataLoader
  loader = torch.utils.data.DataLoader(
      CaptionDataset(args.data_folder, data_name, 'val'),
      batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True)

  hypotheses = list()

  # For each image
  for i, (caps, caplens,img_fea_bef, img_fea_aft,obj_fea_bef,obj_fea_aft,class_bef,class_aft) in enumerate(
          tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
          
    current_index = i

    k = beam_size

    # Move to GPU device, if available
    imgsbef = img_fea_bef.squeeze(1).to(device)
    imgsaft = img_fea_aft.squeeze(1).to(device)
    obj_fea_bef = obj_fea_bef.to(device)
    obj_fea_aft= obj_fea_aft.to(device) 
    class_bef = class_bef.to(device)
    class_aft = class_aft.to(device)
       
    caps = caps.to(device)
    caplens = caplens.to(device)
            
    memory = encoder(imgsbef,imgsaft, obj_fea_bef,obj_fea_aft,class_bef,class_aft)

    tgt = torch.zeros(30,1).to(device).to(torch.int64) ##TODO##
    tgt_length = tgt.size(0)

    mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.to(device)

    tgt[0,0] = word_map['<start>']
    seq = []
    for i in range(tgt_length-1):
      ##
      tgt_embedding = decoder.vocab_embedding(tgt) 
      tgt_embedding = decoder.position_encoding(tgt_embedding) #(length, batch, feature_dim)

      pred = decoder.transformer(tgt_embedding, memory, tgt_mask = mask) #(length, batch, feature_dim)
      pred = decoder.wdc(pred) #(length, batch, vocab_size)

      pred = pred[i,0,:]
      predicted_id = torch.argmax(pred, axis=-1)
   
      ## if word_map['<end>'], end for current sentence
      if predicted_id == word_map['<end>']:
        break

      seq.append(predicted_id)

      ## update mask, tgt
      tgt[i+1,0] = predicted_id
      mask[i+1,0] = 0.0

    # Hypotheses
    temptemp = [w for w in seq if w not in [word_map['<start>'], word_map['<end>'], word_map['<pad>']]]
    hypotheses.append(temptemp)

  #-----------------------------------------------------------------
  kkk = -1
  print (hypotheses)
  for item in hypotheses:
    kkk += 1
    line_hypo = ""

    for word_idx in item:
      #print (word_idx)
      word = get_key(word_map, word_idx)
      #print(word)
      line_hypo += word[0] + " "

    result_json_file[str(kkk)] = []
    result_json_file[str(kkk)].append(line_hypo)

    line_hypo += "\r\n"

  if not os.path.exists('eval_results/'):
    os.mkdir('eval_results/')

  with open('eval_results/' + args.model_name + '_res.json','w') as f:
    json.dump(result_json_file,f)




if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_folder', default='/home/mkhan/stitching/data/track2/')
  parser.add_argument('--checkpoint', default='/home/mkhan/embclip-rearrangement/change_recognition/pano_code/EmbSCU/results/checkpoint_epoch_39_ai2thor_changes_embscu.pth.tar')
  parser.add_argument('--word_map_file', default='/home/mkhan/embclip-rearrangement/change_recognition/pano_code/ai2thor_changes_word2ids.json')
  parser.add_argument('--model_name', default='39new')

  args = parser.parse_args()

  beam_size = 1
  evaluate(args, beam_size)












































