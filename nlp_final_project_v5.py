# If you are awesome start here
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle
# Import data from google drive:

print("Start!")
def get_file(file_name):
    return pickle.load(open('data/new_pickles/' + file_name, 'rb'))

print("load vocab")
en_vocab = get_file('en_vocab.pkl')
de_vocab = get_file('de_vocab.pkl')

print("load rev vocab")
rev_en_vocab = get_file('rev_en_vocab.pkl')
rev_de_vocab = get_file('rev_de_vocab.pkl')

print("load indices")
train_en_indices, train_en_lens = get_file('train_en_data.pkl')
train_de_indices, train_de_lens = get_file('train_de_data.pkl')

#train_en_indices = train_en_indices.to(device)
#train_en_lens = train_en_lens.to(device)
#train_de_indices = train_de_indices.to(device)
#train_de_lens = train_de_lens.to(device)

print(train_en_indices.size())
print(train_de_indices.size())

print(len(en_vocab))
print(len(de_vocab))

#note: train here is only 50 example! do not forget to change this
#num_samples = 100
#train_en_indices = train_en_indices[0:num_samples]
#train_de_indices = train_de_indices[0:num_samples]
#train_en_lens = train_en_lens[0:num_samples]
#train_de_lens = train_de_lens[0:num_samples]

def create_vocab_dicts(data, prev_vocab):
  vocab = {}
  vocab["[CLS]"] = 0
  vocab["[SEP]"] = 1
  vocab["[PAD]"] = 2
  vocab["[UNK]"] = 3
  words = []
  for sentence in data:
    for i in sentence:
      if prev_vocab[int(i)] not in vocab:
        vocab[prev_vocab[int(i)]] = len(vocab)
  return vocab

#en_vocab = create_vocab_dicts(train_en_indices, rev_en_vocab)
#de_vocab = create_vocab_dicts(train_de_indices, rev_de_vocab)

#print(en_vocab)
#print(de_vocab)

# Create reverse vocab dictionaries:
#rev_en_vocab = {v: k for k, v in en_vocab.items()}
#rev_de_vocab = {v: k for k, v in de_vocab.items()}

# print(train_en_indices[indx])
# print(train_de_indices[indx])
# print(train_en_lens[indx])
# print(train_de_lens[indx])

# for en_sent in train_en_indices:
#   print()
#   for word in en_sent:
#     print(rev_en_vocab[int(word)], end=" ")
    
# for de_sent in train_de_indices:
#   print()
#   for word in de_sent:
#     print(rev_de_vocab[int(word)], end=" ")

from torch.utils.data import Dataset, DataLoader

class LangDataset(Dataset):
    """Language dataset dataset."""

    def __init__(self, source_sentences, source_lens, target_sentences, target_lens):
        """
        Args:
            source_data (torch Tensor): Contains all the data
            from the source language represented with indices.
            target_data (torch Tensor): Contains all the data
            from the target language represented with indices.
        """
        self.source_sentences = source_sentences
        self.source_lens = source_lens
        self.target_sentences = target_sentences
        self.target_lens = target_lens

    def __len__(self):
        return len(self.source_lens)

    def __getitem__(self, idx):
        return self.source_sentences[idx], self.source_lens[idx], self.target_sentences[idx], self.target_lens[idx]

import math
class PosEncoder(nn.Module):
    def __init__(self, bsz, max_seq_len, d_emb):  # d_emb is hidden_size
        super(PosEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_emb = d_emb
        
        PE = torch.zeros(max_seq_len, d_emb).to(device)
        for pos in range(0, max_seq_len):
            for i in range(0, int(d_emb/2)):
                PE[pos, 2*i] = math.sin(pos/(10000**(2*i/d_emb)))
                PE[pos, 2*i+1] = math.cos(pos/(10000**(2*i/d_emb)))
        self.PE = PE.repeat(bsz, 1, 1)#.to(device)  # duplicate it by bsz rows

    def forward(self, embs):
        seq_len = embs.size(1)
        bsz = embs.size(0)
        new_embs = embs + self.PE[:bsz, :seq_len, :]
        return new_embs

import copy
class SelfAttention(nn.Module):
    def __init__(self, d_emb, d_k):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(d_emb, d_k).to(device)
        self.Wk = nn.Linear(d_emb, d_k).to(device)
        self.Wv = nn.Linear(d_emb, d_k).to(device)
        
    def forward(self, embs):
        q = self.Wq(embs)
        #print(q.size())
        k = self.Wk(embs)
        #print(type(k))
        v = self.Wv(embs)
        k_transpose = k.permute(0, 2, 1)
        scores = torch.bmm(q, k_transpose)/math.sqrt(self.d_k) 
        scores = torch.nn.functional.softmax(scores, dim=2)
        z = torch.bmm(scores, v)
        #print(z.size())
        return z
      
  
  
class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, d_emb, d_k):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        
        ################ HERE IS THE PROBLEM ###################
        
        self.attns = {}
        for i in range(num_heads):
            self.attns[i] = SelfAttention(d_emb, d_k)
            
#         self.attns = []
#         for _ in range(num_heads):
#             self.attns.append(SelfAttention(d_emb, d_k))
#         self.attns = nn.ModuleList(self.attns)

        ################ END ###################################
        self.W0 = nn.Linear(d_k*num_heads, d_emb).to(device)
        
    def forward(self, embs):
        z = None
        for i in range(self.num_heads):
            inter = self.attns[i](embs)
            if z is None:
                z = inter
            else:
                z = torch.cat((z, inter), dim=2)
        z_new = self.W0(z)
        return z_new

class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        
    def forward(self, x, original_x):
        new_x = x + original_x # residual connection
        norm_x = (new_x - new_x.mean(dim=-1, keepdim=True)) / new_x.std(dim=-1, keepdim=True) # layer normalization
        return norm_x

class FeedForward(nn.Module):
    def __init__(self, d_emb, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear1 = nn.Linear(d_emb, d_ff).to(device)
        self.relu = nn.ReLU().to(device)
        self.linear2 = nn.Linear(d_ff, d_emb).to(device)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, vocab_size, d_emb, d_k, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.multi_attn = MultiheadAttention(num_heads, d_emb, d_k)
        self.add_norm = Add_Norm()
        self.ff = FeedForward(d_emb, d_ff, dropout)
        
        
    def forward(self, embs):
        z = self.multi_attn(embs)
        z_norm = self.add_norm(z, embs)
        z = self.ff(z_norm)
#         z = self.add_norm(z, z_norm)  # or 
        z = self.add_norm(z, embs)
        return z

class TransformerEncoder(nn.Module):
    def __init__(self, bsz, num_layers, num_heads, max_seq_len, vocab_size, d_emb, d_k, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embeddings = nn.Embedding(vocab_size, d_emb).to(device)
        self.pos_en = PosEncoder(bsz, max_seq_len, d_emb)
        self.layers = []
#         for _ in range(num_layers):
#             self.layers.append(EncoderLayer(num_heads, vocab_size, d_emb, d_k, d_ff, dropout))
#         self.layers = nn.ModuleList(self.layers)
        self.layers = {}
        for i in range(num_layers):
            self.layers[i] = EncoderLayer(num_heads, vocab_size, d_emb, d_k, d_ff, dropout)
        
    def forward(self, indices):
        embs = self.embeddings(indices)
        #print(embs.size())
        embs = self.pos_en(embs)
        for i in range(self.num_layers):
            embs = self.layers[i](embs)
        return embs

class AttentionTrain(nn.Module):
  def __init__(self, hidden_size):
    super(AttentionTrain, self).__init__()
    self.hidden_size = hidden_size
    
  def forward(self, enc_output, dec_output):
    enc_output_perm = enc_output.permute(0,2,1)
    #we use batch matrix multiplication to get the attenion scores for each word in each sentence
    batch_attention_scores = torch.bmm(dec_output, enc_output_perm)
    #use batch softmax to get the attention distribution
    batch_attention_dist = torch.nn.functional.softmax(batch_attention_scores, dim=2)
    #get the batch attention outputs
    batch_attn_dist_perm = batch_attention_dist.permute(0,2,1)
    batch_attn_output = torch.bmm(enc_output_perm, batch_attn_dist_perm)
    #make batch_attn_output the same dims at the dec_output
    final_attn_output = batch_attn_output.permute(0,2,1)
    concat_attn_out_and_dec_out = torch.cat((dec_output, final_attn_output), dim=2)
    return concat_attn_out_and_dec_out

class AttentionEval(nn.Module):
  def __init__(self, hidden_size):
    super(AttentionEval, self).__init__()
    self.hidden_size = hidden_size
  
  def forward(self, enc_output, h_n):
    h_n_batch = h_n.unsqueeze(1)

    dec_output = h_n_batch
    enc_output_perm = enc_output.permute(0,2,1)
    #we use batch matrix multiplication to get the attenion scores for each word in each sentence
    batch_attention_scores = torch.bmm(dec_output, enc_output_perm)
    #use batch softmax to get the attention distribution
    batch_attention_dist = torch.nn.functional.softmax(batch_attention_scores, dim=2)
    #get the batch attention outputs
    batch_attn_dist_perm = batch_attention_dist.permute(0,2,1)
    batch_attn_output = torch.bmm(enc_output_perm, batch_attn_dist_perm)
    #make batch_attn_output the same dims at the dec_output
    final_attn_output = batch_attn_output.permute(0,2,1)
    concat_attn_out_and_dec_out = torch.cat((dec_output, final_attn_output), dim=2)
    return concat_attn_out_and_dec_out

class Dec_Att_Layer_train(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Dec_Att_Layer_train, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True).to(device)
        self.attn_train = AttentionTrain(hidden_size).to(device)
        self.linear = nn.Linear(hidden_size*2, hidden_size).to(device)
        
    def forward(self, embs, enc_output):
        dec_output, (h_n, c_n) = self.lstm(embs)
        attn_output = self.attn_train(enc_output, dec_output)
        layer_output = self.linear(attn_output)
        return layer_output

class Dec_Att_Layer_eval(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Dec_Att_Layer_eval, self).__init__()
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size, bias=True).to(device)
        self.attn_eval = AttentionEval(hidden_size).to(device)
        self.linear = nn.Linear(hidden_size*2, hidden_size).to(device)
        
    def forward(self, embs, h_n, c_n, enc_output):
        h_n, c_n = self.lstm_cell(embs, (h_n, c_n))
        h_n_attn = self.attn_eval(enc_output, h_n)
#         print(h_n_attn.size())
        layer_output = self.linear(h_n_attn)
        return layer_output, h_n, c_n

# decoder I used to test Trasnformer encoder, basically just got rid of h0, c0
class DecoderAttention3(nn.Module):
  def __init__(self, num_layers, vocab_size, hidden_size):
    super(DecoderAttention3, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    
    self.embeddings = nn.Embedding(vocab_size, hidden_size)
#     self.dec_layers_train = nn.ModuleList([Dec_Att_Layer_train(vocab_size, hidden_size) for _ in range(num_layers)])
    self.dec_layers_train = {}
    self.dec_layers_eval = {}
    for i in range(num_layers):
        self.dec_layers_train[i] = Dec_Att_Layer_train(vocab_size, hidden_size)
        self.dec_layers_eval[i] = Dec_Att_Layer_eval(vocab_size, hidden_size)
#     self.dec_layers_eval = nn.ModuleList([Dec_Att_Layer_eval(vocab_size, hidden_size) for _ in range(num_layers)])
    self.linear = nn.Linear(hidden_size, vocab_size, bias=True)
    
    for i in range(num_layers):
        self.dec_layers_eval[i].lstm_cell.weight_ih = self.dec_layers_train[i].lstm.weight_ih_l0
        self.dec_layers_eval[i].lstm_cell.weight_hh = self.dec_layers_train[i].lstm.weight_hh_l0
        self.dec_layers_eval[i].lstm_cell.bias_ih = self.dec_layers_train[i].lstm.bias_ih_l0
        self.dec_layers_eval[i].lstm_cell.bias_hh = self.dec_layers_train[i].lstm.bias_hh_l0
        self.dec_layers_eval[i].linear.weight = self.dec_layers_train[i].linear.weight
        self.dec_layers_eval[i].linear.bias = self.dec_layers_train[i].linear.bias
    
    
  def forward(self, indices, enc_output):  
    
    if self.training == False:
      max_seq_len = indices.size(1)
      bsz = indices.size(0)
      next_word_idx = torch.zeros([bsz,1]).long().to(device) # Make the first LSTM input [CLS]
      eos_flags = torch.zeros(bsz).long().to(device)
      scores = None
#       scores = []
      i = 0      
      h_n = {}
      c_n = {}
      while int(torch.sum(eos_flags)) < bsz:
        # LSTMCell expects and input (batch, input_size), squeeze seq_len
        embs = self.embeddings(next_word_idx).squeeze(1)
        #print('Emb size: ',  embs.size())
        if i == 0:
            h_0 = torch.zeros(bsz, self.hidden_size).to(device)
            c_0 = torch.zeros(bsz, self.hidden_size).to(device)
            for j in range(self.num_layers):
                embs, h_n[j], c_n[j] = self.dec_layers_eval[j](embs, h_0, c_0, enc_output)
                embs = embs.squeeze(1)
            #print('New emb size: ', embs.size())
        else:
            for j in range(self.num_layers):
                embs, h_n[j], c_n[j] = self.dec_layers_eval[j](embs, h_n[j], c_n[j], enc_output)
                embs = embs.squeeze(1)

#         score = self.linear(embs)
#         next_word_idx = torch.argmax(score.squeeze(1), dim=1)
#         scores.append(score)

        if scores is None:
          scores = self.linear(embs.unsqueeze(1))
          next_word_idx = torch.argmax(scores.squeeze(1), dim=1)
        else:
          scores_n = self.linear(embs.unsqueeze(1))
          next_word_idx = torch.argmax(scores_n.squeeze(1), dim=1)
          scores = torch.cat((scores, scores_n), dim=1)
        
        # Doing greedy search by getting the position of the highest value in the squence
        # next_word_idx = torch.argmax(scores[:, -1, :].squeeze(1), dim=1)
        # Check if there are EOS, so in our case [SEP]. Index is 1
        #############
        #next_word_idx = indices[:, i+1]
        
        flags_to_add = (next_word_idx[:] == 1).long()
        eos_flags += flags_to_add
        # Just in case an [SEP] is generated twice
        eos_flags[eos_flags > 1] = 1
        
        # Unsqueezing an extra dimension to match the input for the next iteration
        next_word_idx = next_word_idx.unsqueeze(1)
        if i > max_seq_len-2:
          #print("Sentence is too long:", i)
          break
        i+=1
      #print('Eval mode: ', len(scores[0]))
#       scores = torch.cat(scores, dim=1)
      #print(scores.size())
      return scores
    
    
    if (self.training == True): #model is in training mode
      embs = self.embeddings(indices)
      for i in range(self.num_layers):
          embs = self.dec_layers_train[i](embs, enc_output)
      scores = self.linear(embs)
      
#       dec_output, (h_n, c_n) = self.lstm(embeddings)    
#       attn_output = self.attention_train(enc_output, dec_output)
    
#       scores = self.linear(attn_output)
      #print('Train mode: ', scores.size())
      return scores

# train function used to test Transformer encoder
def train(lang_dataset, params, encoder, decoder):
    
    # since the second index corresponds to the PAD token, we just ignore it
    # when computing the loss
    criterion = nn.CrossEntropyLoss()#ignore_index=2)
    
    optim_encoder = optim.Adam(encoder.parameters(), lr=params['learning_rate'])
    optim_decoder = optim.Adam(decoder.parameters(), lr=params['learning_rate'])

    dataloader = DataLoader(lang_dataset, batch_size=params['batch_size'], shuffle=True)
    
    for epoch in range(params['epochs']):
        ep_loss = 0.
        accuracy = 0.
        start_time = time.time()
        # for each batch, calculate loss and optimize model parameters            
        for de_indices, de_lens, en_indices, en_lens in dataloader:            
            # Remove pads based on largest sentence in batch
            max_de_len = torch.max(de_lens).to(device)
            de_indices = de_indices[:, :max_de_len].to(device)
            max_en_len = torch.max(en_lens).to(device)
            en_indices = en_indices[:, :max_en_len].to(device)
            
            enc_outputs = encoder(de_indices)
            preds = decoder(en_indices[:,0:-1],enc_outputs)
            #print('Preds full: ', torch.argmax(preds[5], dim=1))
            
            targets = en_indices[:,1:]
            #print('Target full: ', targets[5])
            
            preds = preds.permute(0, 2, 1)
#             print(preds.size())
            preds_index = torch.argmax(preds, dim=1)
#             print(preds_index[0].size())
#             for j in range(len(preds_index[0])):
#               print(preds_index[0,j])

            
            targets = targets[:,0:preds.size()[2]]
            loss = criterion(preds, targets)
        
            loss /= params['batch_size']
            
            loss.backward()
            optim_encoder.step()
            optim_encoder.zero_grad()
            optim_decoder.step()
            optim_decoder.zero_grad()
            
            correct = (preds_index == targets).sum().float()
            accuracy += float(correct) / (params['batch_size'] * preds.size()[2])
            
            ep_loss += float(loss)
    
        with open("loss.txt", "a+") as l:
            l.write("Epoch: " + str(epoch) + " Loss: " + str(ep_loss) + " time: " + str(time.time()-start_time) + " Accuracy: " + str(accuracy / len(dataloader)) + "\n")
        
        if epoch % 5 == 0:
            torch.save(encoder, './models/t_encoder_' + str(params['hidden_size']) + '_' + str(epoch) + '.pt')
            torch.save(decoder, './models/t_decoder_' + str(params['hidden_size']) + '_' + str(epoch) + '.pt')

params = {}
params['batch_size'] = 50
params['epochs'] = 100
params['learning_rate'] = 0.01
params['hidden_size'] = 512

params['num_layers'] = 3
params['num_layers_dec'] = 3
params['num_heads'] = 3
params['max_seq_len'] = 1000

params['keys_dim'] = 64
params['feed_forward_dim'] = 2048
params['dropout'] = 0.001

# encoder = Encoder(len(de_vocab), hidden_size)
# decoder = Decoder(len(en_vocab), hidden_size)

# encoder.eval()
# decoder.eval()

encoder = TransformerEncoder(params['batch_size'], params['num_layers'], 
                             params['num_heads'], params['max_seq_len'], 
                             len(de_vocab), params['hidden_size'], 
                             params['keys_dim'], params['feed_forward_dim'], 
                             params['dropout'])

#encoder = nn.DataParallel(encoder)
#encoder.to(device)

decoder = DecoderAttention3(params['num_layers_dec'], len(en_vocab), params['hidden_size']).to(device)

#decoder = nn.DataParallel(decoder)
#decoder.to(device)

lang_dataset = LangDataset(train_de_indices, train_de_lens, train_en_indices, train_en_lens)


# encoder.load_state_dict(torch.load(root_folder + 'Models/Transformer/t_encoder_512_499.pt'))
# decoder.load_state_dict(torch.load(root_folder + 'Models/Transformer/t_decoder_512_499.pt'))

train(lang_dataset, params, encoder, decoder)
print()

