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
data_name = 'ai2thor_changes'

# Model parameters 
embed_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Training parameters
start_epoch = 0
batch_size = 4#16
workers = 0
decoder_lr = 1e-4
encoder_lr = 1e-4
grap_clip = 5.
best_bleu4 = 0.
print_freq = 100

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

  encoder = MCCFormers_S(args.feature_dim,h=64,w=64, n_head=args.n_head,n_layers=args.n_layers).to(device)
  encoder_four = MCCFormers_S_four(feature_dim = 2048,h=16,w=16, n_head=args.n_head,n_layers=args.n_layers).to(device)  
  decoder = DecoderTransformer(feature_dim = args.feature_dim_de,
                               vocab_size = len(word_map),
                               n_head = args.n_head,
                               n_layers = args.n_layers,
                               dropout=dropout).to(device)


  
  encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                       lr=encoder_lr)

  encoder_four_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_four.parameters()),
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
  pbar2 = tqdm(total=len(train_loader), desc=f"Process 1", leave=False)
  start = time.time()

  # Batches
  for i, (imgs0_0, imgs0_1, caps, caplens) in enumerate(train_loader):
    data_time.update(time.time() - start)

    # Move to GPU, if available
    imgs0_0 = imgs0_0.to(device)
    imgs0_1 = imgs0_1.to(device)
    imgs0_0 = res101(imgs0_0)
    imgs0_1 = res101(imgs0_1) 
     
    
    caps = caps.to(device)
    caplens = caplens.to(device)
  
    
    
    # Forward prop.
    l = encoder(imgs0_0, imgs0_1)
    #l1 = encoder(imgs1_0, imgs1_1)
    #l2 = encoder(imgs2_0, imgs2_1)
    #l3 = encoder(imgs3_0, imgs3_1)        
    #l = encoder_four(imgs0_0, imgs0_1)
    
    #print(l0.size())
    #print(l.size())
    
    #scores0, caps_sorted0, decode_lengths0, sort_ind0 = decoder(l0, cap0, caplen0)    
    #scores1, caps_sorted1, decode_lengths1, sort_ind1 = decoder(l1, cap1, caplen1)
    #scores2, caps_sorted2, decode_lengths2, sort_ind2 = decoder(l2, cap2, caplen2)    
    #scores3, caps_sorted3, decode_lengths3, sort_ind3 = decoder(l3, cap3, caplen3)     
    scores, caps_sorted, decode_lengths, sort_ind = decoder(l, caps, caplens)

    #targets0 = caps_sorted0[:, 1:]    
    #targets1 = caps_sorted1[:, 1:]       
    #targets2 = caps_sorted2[:, 1:]   
    #targets3 = caps_sorted3[:, 1:]           
    targets = caps_sorted[:, 1:]

    #scores0 = pack_padded_sequence(scores0, decode_lengths0, batch_first=True).data
    #targets0 = pack_padded_sequence(targets0, decode_lengths0, batch_first=True).data
    #scores1 = pack_padded_sequence(scores1, decode_lengths1, batch_first=True).data
    #targets1 = pack_padded_sequence(targets1, decode_lengths1, batch_first=True).data
    #scores2 = pack_padded_sequence(scores2, decode_lengths2, batch_first=True).data
    #targets2 = pack_padded_sequence(targets2, decode_lengths2, batch_first=True).data
    #scores3 = pack_padded_sequence(scores3, decode_lengths3, batch_first=True).data
    #targets3 = pack_padded_sequence(targets3, decode_lengths3, batch_first=True).data    
        
    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data


    loss = criterion(scores, targets)# + criterion(scores0, targets0) + criterion(scores1, targets1) + criterion(scores2, targets2) + criterion(scores3, targets3)

    # Back prop.
    #encoder_optimizer.zero_grad()
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
    pbar2.update(1)
    start = time.time()
    
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
  parser.add_argument('--feature_dim', type=int, default=2048)
  parser.add_argument('--feature_dim_de', type=int, default=512)


  args = parser.parse_args()

  main(args)
