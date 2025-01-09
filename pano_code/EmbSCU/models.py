import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MCCFormers_S(nn.Module):
  """
  MCCFormers-S
  """

  def __init__(self, feature_dim, h, w, d_model = 512, n_head = 4, n_layers = 2, dim_feedforward = 2048):
    """
    :param feature_dim: feature dimension of input dimension
    :param d_model: dimension of input to Transformer
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    :param dim_feedforward: dimension of hidden state
    :param h: height of input image
    :param w: width of input image
    """
    super(MCCFormers_S, self).__init__()

    self.input_proj = nn.Conv2d(feature_dim, d_model, kernel_size = 1)
    self.linear_layer = nn.Linear(feature_dim, d_model)
    self.linear_layer2 = nn.Linear(2, d_model)
    self.d_model = d_model

    encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward = dim_feedforward)
    self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    self.idx_embedding = nn.Embedding(2, d_model)
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))
    self.embedding_layer = nn.Embedding(49, d_model) #48 object classes + None
  def forward(self, img_feat1, img_feat2, obj_fea_bef, obj_fea_aft,class_bef,class_aft):
    # img_feat1 (batch_size, feature_dim, h, w)
    #print("in models", img_feat1.shape, img_feat2.shape, obj_fea_bef.shape, obj_fea_aft.shape, class_bef.shape,class_aft.shape) #([2, 256, 64, 64]) ([2, 20, 256])
    batch = img_feat1.size(0)
    #print (batch)

    #h, w = panorama_shape[0], panorama_shape[1]


    obj_fea_bef = self.linear_layer(obj_fea_bef).permute(1, 0, 2)
    obj_fea_aft = self.linear_layer(obj_fea_aft).permute(1, 0, 2)

    class_bef = self.embedding_layer(class_bef).permute(1, 0, 2)
    class_aft = self.embedding_layer(class_aft).permute(1, 0, 2)
    #print (obj_fea_bef.shape, class_aft.shape)
    img_feat1 = self.linear_layer2(img_feat1.float()).permute(1, 0, 2)
    #print ("img_feat1.shape", img_feat1.shape)
    img_feat2 = self.linear_layer2(img_feat2.float()).permute(1, 0, 2)
    #print (img_feat1.shape) 


    #print ("img_feat1 bef", img_feat1.shape, obj_fea_bef.shape)
    img_feat1 += obj_fea_bef 
    img_feat1 += class_bef # (batch, d_model, h*w)
    img_feat2 += obj_fea_aft
    img_feat2 += class_aft  # (batch, d_model, h*w)
    #print ("img_feat1 aft", img_feat1.shape)
    img_feat_cat = torch.cat([img_feat1, img_feat2], dim=0)  # (batch, d_model, 2*h*w)
    #img_feat_cat = img_feat_cat.permute(2, 0, 1)  # (2*h*w, batch, d_model)
    #print (img_feat_cat.shape)
    # Create index tensor for idx_embedding
    idx1 = torch.zeros(batch, 20).long().to(device)
    idx2 = torch.ones(batch, 20).long().to(device)
    idx = torch.cat([idx1, idx2], dim=1)  # (batch, 2*h*w)

    
    # Get index embeddings
    idx_embedding = self.idx_embedding(idx)  # (batch, 2*h*w, d_model)
    idx_embedding = idx_embedding.permute(1, 0, 2)  # (2*h*w, batch, d_model)
    #print (idx_embedding.shape)
    # Combine image features and index embeddings
    feature = img_feat_cat + idx_embedding  # (2*h*w, batch, d_model) 
    #print ("features", feature.shape) #([8192, 4, 512])

    #feature = torch.cat([feature, obj_fea_bef, obj_fea_aft, class_bef, class_aft], dim=0)  # Adjust dimension as necessary  #([8192, 4, 512]) ([163840, 4, 512])

    feature = self.transformer(feature)  # (2*h*w + combined_features, batch, d_model) 8192 + combined_features, 4, 512 
    #print ("feature end", feature.shape)
    return feature
  
  def forward2(self, img_feat1, img_feat2, obj_fea_bef, obj_fea_aft,class_bef,class_aft):
    # img_feat1 (batch_size, feature_dim, h, w)
    #print(img_feat1.shape, img_feat2.shape, obj_fea_bef.shape, obj_fea_aft.shape, class_bef.shape,class_aft.shape) #([2, 256, 64, 64]) ([2, 20, 256])
    batch = img_feat1.size(0)
    
    feature_dim = img_feat1.size(1)
    h, w = img_feat1.size(2), img_feat1.size(3)


    d_model = self.d_model
    obj_fea_bef = self.linear_layer(obj_fea_bef).permute(1, 0, 2)
    obj_fea_aft = self.linear_layer(obj_fea_aft).permute(1, 0, 2)
    class_bef = self.embedding_layer(class_bef).permute(1, 0, 2)
    class_aft = self.embedding_layer(class_aft).permute(1, 0, 2)
    #print (obj_fea_bef.shape, class_aft.shape)
    img_feat1 = self.input_proj(img_feat1)
    img_feat2 = self.input_proj(img_feat2)
    #print (img_feat1.shape) 


    img_feat1 = img_feat1.view(batch, d_model, -1)  # (batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1)  # (batch, d_model, h*w)
    #print("here", img_feat1.shape, img_feat2.shape) #([2, 512, 4096])

    # Position embedding
    pos_w = torch.arange(w, device=device)
    pos_h = torch.arange(h, device=device)

    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([
        embed_w.unsqueeze(0).repeat(h, 1, 1),
        embed_h.unsqueeze(1).repeat(1, w, 1)], dim=-1) 
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)  # (batch, d_model, h, w)
    position_embedding = position_embedding.view(batch, d_model, -1)  # (batch, d_model, h*w)

    img_feat1 += position_embedding  # (batch, d_model, h*w)
    img_feat2 += position_embedding  # (batch, d_model, h*w)

    img_feat_cat = torch.cat([img_feat1, img_feat2], dim=2)  # (batch, d_model, 2*h*w)
    img_feat_cat = img_feat_cat.permute(2, 0, 1)  # (2*h*w, batch, d_model)

    # Create index tensor for idx_embedding
    idx1 = torch.zeros(batch, h * w).long().to(device)
    idx2 = torch.ones(batch, h * w).long().to(device)
    idx = torch.cat([idx1, idx2], dim=1)  # (batch, 2*h*w)

    
    # Get index embeddings
    idx_embedding = self.idx_embedding(idx)  # (batch, 2*h*w, d_model)
    idx_embedding = idx_embedding.permute(1, 0, 2)  # (2*h*w, batch, d_model)

    # Combine image features and index embeddings
    feature = img_feat_cat + idx_embedding  # (2*h*w, batch, d_model) 
    #print ("features", feature.shape) #([8192, 4, 512])

    feature = torch.cat([feature, obj_fea_bef, obj_fea_aft, class_bef, class_aft], dim=0)  # Adjust dimension as necessary  #([8192, 4, 512]) ([163840, 4, 512])

    feature = self.transformer(feature)  # (2*h*w + combined_features, batch, d_model) 8192 + combined_features, 4, 512 
    #print ("feature end", feature.shape)
    return feature


  def forward2(self, img_feat1, img_feat2, obj_fea_bef,obj_fea_aft,class_bef,class_aft):
    # img_feat1 (batch_size, feature_dim, h, w)
    #print (img_feat1.shape, img_feat2.shape, obj_fea_bef.shape,obj_fea_aft.shape,class_bef.shape,class_aft.shape)
    batch = img_feat1.size(0)
    
    feature_dim = img_feat1.size(1)
    h, w = img_feat1.size(2), img_feat1.size(3)
    #print ("")

    #print ("here",batch, h, w)
    #print ("")

    d_model = self.d_model

    img_feat1 = self.input_proj(img_feat1)
    img_feat2 = self.input_proj(img_feat2)
    img_feat1 = img_feat1.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1) #(batch, d_model, h*w)
    #print ("here",img_feat1.shape, img_feat2.shape)
    # position embedding
    pos_w = torch.arange(w, device=device)#.to(device)
    pos_h = torch.arange(h, device=device)#.to(device)

    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                    embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                    dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)
    position_embedding = position_embedding.view(batch, d_model, -1)
    #print ("position_embedding",position_embedding.shape)
    img_feat1 = img_feat1 + position_embedding #(batch, d_model, h*w)
    img_feat2 = img_feat2 + position_embedding #(batch, d_model, h*w)

    img_feat_cat = torch.cat([img_feat1, img_feat2], dim = 2) #(batch, d_model, 2*h*w)
    img_feat_cat = img_feat_cat.permute(2, 0, 1) #(2*h*w, batch, d_model)

    # idx = 0, 1 for img_feat1, img_feat2, respectively
    #idx1 = torch.zeros(batch, h*w)
    idx1 = torch.zeros(batch, h*w).long()#.to(device)
    #print (idx1.shape)
    idx2 = torch.ones(batch, h*w).long()#.to(device)
    #print (idx2.shape)
    idx = torch.cat([idx1, idx2], dim = 1)#.to(device) #(batch, 2*h*w)
    #print (idx.shape, idx)
    idx_embedding = self.idx_embedding(idx.to(device)) #(batch, 2*h*w, d_model)
    idx_embedding = idx_embedding.permute(1, 0, 2) #(2*h*w, batch, d_model)

    feature = img_feat_cat + idx_embedding #(2*h*w, batch, d_model)
    feature = self.transformer(feature) #(2*h*w, batch, d_model)
    
    '''
    img_feat1 = feature[:h*w].permute(1, 2, 0)  #(batch, d_model, h*w)
    img_feat1 = img_feat1.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)
    img_feat2 = feature[h*w:].permute(1, 2, 0)  #(batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)
    
    img_feat = torch.cat([img_feat1,img_feat2],dim=2)
    '''
    
    return feature
    
class MCCFormers_S1(nn.Module):
  """
  MCCFormers-S
  """

  def __init__(self, feature_dim, h, w, d_model = 512, n_head = 4, n_layers = 2, dim_feedforward = 2048):
    """
    :param feature_dim: feature dimension of input dimension
    :param d_model: dimension of input to Transformer
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    :param dim_feedforward: dimension of hidden state
    :param h: height of input image
    :param w: width of input image
    """
    super(MCCFormers_S1, self).__init__()

    self.input_proj = nn.Conv2d(feature_dim, d_model, kernel_size = 1)

    self.d_model = d_model

    encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward = dim_feedforward)
    self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    self.idx_embedding = nn.Embedding(2, d_model)
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))

  def forward(self, img_feat1, img_feat2):
    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    h, w = img_feat1.size(2), img_feat1.size(3)

    d_model = self.d_model

    img_feat1 = self.input_proj(img_feat1)
    img_feat2 = self.input_proj(img_feat2)
    img_feat1 = img_feat1.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1) #(batch, d_model, h*w)

    # position embedding
    pos_w = torch.arange(w, device=device).to(device)
    pos_h = torch.arange(h, device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                    embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                    dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)
    position_embedding = position_embedding.view(batch, d_model, -1)

    img_feat1 = img_feat1 + position_embedding #(batch, d_model, h*w)
    img_feat2 = img_feat2 + position_embedding #(batch, d_model, h*w)

    img_feat_cat = torch.cat([img_feat1, img_feat2], dim = 2) #(batch, d_model, 2*h*w)
    img_feat_cat = img_feat_cat.permute(2, 0, 1) #(2*h*w, batch, d_model)

    # idx = 0, 1 for img_feat1, img_feat2, respectively
    idx1 = torch.zeros(batch, h*w).long().to(device)
    idx2 = torch.ones(batch, h*w).long().to(device)
    idx = torch.cat([idx1, idx2], dim = 1) #(batch, 2*h*w)
    idx_embedding = self.idx_embedding(idx) #(batch, 2*h*w, d_model)
    idx_embedding = idx_embedding.permute(1, 0, 2) #(2*h*w, batch, d_model)

    feature = img_feat_cat + idx_embedding #(2*h*w, batch, d_model)
    feature = self.transformer(feature) #(2*h*w, batch, d_model)
    
    
    img_feat1 = feature[:h*w].permute(1, 2, 0)  #(batch, d_model, h*w)
    img_feat1 = img_feat1.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)
    img_feat2 = feature[h*w:].permute(1, 2, 0)  #(batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)
    
    img_feat = torch.cat([img_feat1,img_feat2],dim=2)
    
    
    return img_feat
    
        
class MCCFormers_S_four(nn.Module):
  """
  MCCFormers-S
  """

  def __init__(self, feature_dim, h, w, d_model = 512, n_head = 4, n_layers = 2, dim_feedforward = 2048):
    """
    :param feature_dim: feature dimension of input dimension
    :param d_model: dimension of input to Transformer
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    :param dim_feedforward: dimension of hidden state
    :param h: height of input image
    :param w: width of input image
    """
    super(MCCFormers_S_four, self).__init__()

    self.input_proj = nn.Conv2d(feature_dim, d_model, kernel_size = 1)

    self.d_model = d_model

    encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward = dim_feedforward)
    self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    self.idx_embedding = nn.Embedding(8, d_model)
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))

  def forward(self, img_feat1, img_feat2, img_feat3, img_feat4, img_feat5, img_feat6, img_feat7, img_feat8):
    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    h, w = img_feat1.size(2), img_feat1.size(3)

    d_model = self.d_model

    img_feat1 = self.input_proj(img_feat1)
    img_feat2 = self.input_proj(img_feat2)
    img_feat3 = self.input_proj(img_feat3)
    img_feat4 = self.input_proj(img_feat4)    
    img_feat5 = self.input_proj(img_feat5)
    img_feat6 = self.input_proj(img_feat6)    
    img_feat7 = self.input_proj(img_feat7)
    img_feat8 = self.input_proj(img_feat8)    
    
    img_feat1 = img_feat1.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat3 = img_feat3.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat4 = img_feat4.view(batch, d_model, -1) #(batch, d_model, h*w)    
    img_feat5 = img_feat5.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat6 = img_feat6.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat7 = img_feat7.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat8 = img_feat8.view(batch, d_model, -1) #(batch, d_model, h*w)      

    # position embedding
    pos_w = torch.arange(w, device=device).to(device)
    pos_h = torch.arange(h, device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                    embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                    dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)
    position_embedding = position_embedding.view(batch, d_model, -1)

    img_feat1 = img_feat1 + position_embedding #(batch, d_model, h*w)
    img_feat2 = img_feat2 + position_embedding #(batch, d_model, h*w)
    img_feat3 = img_feat3 + position_embedding #(batch, d_model, h*w)    
    img_feat4 = img_feat4 + position_embedding #(batch, d_model, h*w)
    img_feat5 = img_feat5 + position_embedding #(batch, d_model, h*w)      
    img_feat6 = img_feat6 + position_embedding #(batch, d_model, h*w)    
    img_feat7 = img_feat7 + position_embedding #(batch, d_model, h*w)
    img_feat8 = img_feat8 + position_embedding #(batch, d_model, h*w)          

    img_feat_cat = torch.cat([img_feat1, img_feat2, img_feat3, img_feat4, img_feat5, img_feat6, img_feat7, img_feat8], dim = 2) #(batch, d_model, 8*h*w)
    img_feat_cat = img_feat_cat.permute(2, 0, 1) #(8*h*w, batch, d_model)

    
    idx = torch.arange(8, device=device).to(device) # (1, 8)

    idx_embedding = self.idx_embedding(idx)  # (8, d_model) 
    idx_embedding = idx_embedding.unsqueeze(1).unsqueeze(1) # (8, 1 , 1,  d_model) 
    idx_embedding = idx_embedding.repeat(1,h*w, 1, 1)
    idx_embedding = idx_embedding.repeat(1,1, batch, 1)
    idx_embedding = idx_embedding.view(8*h*w, batch, d_model) #(8*h*w, batch, d_model)
    
    feature = img_feat_cat + idx_embedding #(8*h*w, batch, d_model)
    
    feature = self.transformer(feature) #(8*h*w, batch, d_model)


    return feature



class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, max_len=5000):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)

      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(0).transpose(0, 1)
      self.register_buffer('pe', pe)

  def forward(self, x):
      x = x + self.pe[:x.size(0), :]
      return self.dropout(x)

class DecoderTransformer(nn.Module):
  """
  Decoder with Transformer.
  """

  def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
    """
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    """
    super(DecoderTransformer, self).__init__()

    self.feature_dim = feature_dim
    self.embed_dim = feature_dim
    self.vocab_size = vocab_size
    self.dropout = dropout

    # embedding layer
    self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim) #vocaburaly embedding
    
    # Transformer layer
    decoder_layer = nn.TransformerDecoderLayer(feature_dim, n_head, dim_feedforward = feature_dim * 4, dropout=self.dropout)
    self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
    self.position_encoding = PositionalEncoding(feature_dim)
    
    # Linear layer to find scores over vocabulary
    self.wdc = nn.Linear(feature_dim, vocab_size)
    self.dropout = nn.Dropout(p=self.dropout)
    self.init_weights() # initialize some layers with the uniform distribution

  def init_weights(self):
    """
    Initializes some parameters with values from the uniform distribution, for easier convergence
    """
    self.vocab_embedding.weight.data.uniform_(-0.1,0.1)

    self.wdc.bias.data.fill_(0)
    self.wdc.weight.data.uniform_(-0.1,0.1)    
 

  def forward(self, memory, encoded_captions, caption_lengths):
    """
    :param memory: image feature (S, batch, feature_dim)
    :param tgt: target sequence (length, batch)
    :param sentence_index: sentence index of each token in target sequence (length, batch)
    """
    #memory = torch.cat([memory1,memory2],dim=2)

    tgt = encoded_captions.permute(1,0)
    tgt_length = tgt.size(0)

    mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.to(device)

    tgt_embedding = self.vocab_embedding(tgt) 
    tgt_embedding = self.position_encoding(tgt_embedding) #(length, batch, feature_dim)

    #print(memory.size())
    
    #print(tgt_embedding.size())

    pred = self.transformer(tgt_embedding, memory, tgt_mask = mask) #(length, batch, feature_dim)
    pred = self.wdc(self.dropout(pred)) #(length, batch, vocab_size)

    pred = pred.permute(1,0,2)

    # Sort input data by decreasing lengths
    caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
    encoded_captions = encoded_captions[sort_ind]
    pred = pred[sort_ind]
    decode_lengths = (caption_lengths - 1).tolist()

    return pred, encoded_captions, decode_lengths, sort_ind

