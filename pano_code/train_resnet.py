import json
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import MCCFormers_S, MCCFormers_S_four, DecoderTransformer
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu ##--

from torch.autograd import Variable
import torchvision.models as models
import argparse

# Data parameters
data_name = 'ai2thor_changes_imgfeaonly'

# Model parameters 
embed_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Training parameters
start_epoch = 0
batch_size = 8
workers = 0
decoder_lr = 1e-4
encoder_lr = 1e-4
grap_clip = 5.
best_bleu4 = 0.
print_freq = 10

para_lambda1 = 1.0
para_lambda2 = 1.0

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def main(args):

  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.benckmark = False
  torch.backends.cudnn.deterministic = True

  global start_epoch, data_name

  # Read word map
  word_map_file = '/home/mkhan/embclip-rearrangement/change_recognition/pano_code/ai2thor_changes_word2ids.json'
  with open(word_map_file, 'r') as f:
    word_map = json.load(f)

  # Initialize

  encoder = MCCFormers_S(feature_dim = args.feature_dim,h=64,w=64, n_head=args.n_head,n_layers=args.n_layers).to(device) 
  decoder = DecoderTransformer(feature_dim = args.feature_dim_de,
                               vocab_size = len(word_map),
                               n_head = args.n_head,
                               n_layers = args.n_layers,
                               dropout=dropout).to(device)


  
  encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                       lr=encoder_lr)

  decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                       lr=decoder_lr)


  #image_backbone = models.resnet101(pretrained=True)
  #features = list(backbone.children())[:-2]
  #model_image_o = nn.Sequential(*features)
  
  backbone1 = models.resnet101(pretrained=True)
  features1 = list(backbone1.children())[:-2]
  res101 = nn.Sequential(*features1).cuda().to(device)  

  criterion = nn.CrossEntropyLoss().to(device)

  train_loader = torch.utils.data.DataLoader(
    CaptionDataset(args.data_folder, data_name, 'train'),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
  
  for epoch in range(start_epoch, args.epochs):
    print("epoch : " + str(epoch))

    train(train_loader=train_loader,
          res101=res101,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,        
          decoder_optimizer=decoder_optimizer,
          epoch=epoch,
          word_map=word_map
          )

    # Save checkpoint
    save_checkpoint(args.root_dir, data_name, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)
    

def train(train_loader, res101, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, word_map):
  
  encoder.train()
  decoder.train()
  res101.eval()


  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top3accs = AverageMeter()

  start = time.time()
  pbar2 = tqdm(total=len(train_loader), desc=f"Process 1", leave=False)
  # Batches
  for i, (caps, caplens, img_fea_bef, img_fea_aft,obj_fea_bef,obj_fea_aft,class_bef,class_aft) in enumerate(train_loader):
    data_time.update(time.time() - start)

    # Move to GPU, if available ###TODO -- CHANGE THE SHAPE ACCORDINGLY (dimension_x, dimension_y, dimension_z) --- Change the shape of encoder (accordingly) 

    imgsbef = img_fea_bef.squeeze(1).to(device)
    imgsaft = img_fea_aft.squeeze(1).to(device)
       

    caps = caps.to(device)
    caplens = caplens.to(device)
    obj_fea_bef = obj_fea_bef.to(device)
    obj_fea_aft= obj_fea_aft.to(device) 
    class_bef = class_bef.to(device)
    class_aft = class_aft.to(device)
    
    
    # Forward prop.
      
    l = encoder(imgsbef,imgsaft, obj_fea_bef,obj_fea_aft,class_bef,class_aft) ### change shape ----
    
  
    scores, caps_sorted, decode_lengths, sort_ind = decoder(l, caps, caplens) #

      
    targets = caps_sorted[:, 1:]

        
    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data


    loss = criterion(scores, targets)

    # Back prop.
    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()


    # Keep track of metrics
    top3 = accuracy(scores, targets, 3)
    losses.update(loss.item(), sum(decode_lengths))
    top3accs.update(top3, sum(decode_lengths))
    batch_time.update(time.time() - start)

    start = time.time()
    pbar2.update(1)
    # Print status
    if i % print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time,
                                                                    loss=losses,
                                                                    top3=top3accs))
  pbar2.close()

if __name__=='__main__':
  
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_folder', default='/home/mkhan/stitching/data/track2/')
  parser.add_argument('--root_dir', default='results/')
  parser.add_argument('--hidden_dim', type=int, default=512)
  parser.add_argument('--attention_dim', type=int, default=512)
  parser.add_argument('--epochs', type=int, default=40)
  parser.add_argument('--n_head', type=int, default=4)
  parser.add_argument('--n_layers', type=int, default=2)
  parser.add_argument('--feature_dim', type=int, default=256)#2048#
  parser.add_argument('--feature_dim_de', type=int, default=512)


  args = parser.parse_args()
  
  main(args)
