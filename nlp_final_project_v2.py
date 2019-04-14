# If you are awesome start here
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import time
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Start!")
def get_file(file_name):
    return pickle.load(open('data/' + file_name, 'rb'))

print("load vocab")
en_vocab = get_file('en_vocab.pkl')
de_vocab = get_file('de_vocab.pkl')

print("load rev vocab")
rev_en_vocab = get_file('rev_en_vocab.pkl')
rev_de_vocab = get_file('rev_de_vocab.pkl')

print("load indices")
train_en_indices = get_file('train_en_indices.pkl')
train_de_indices = get_file('train_de_indices.pkl')

print("done!")

print(train_en_indices.size())
print(train_de_indices.size())

print(train_en_indices[1])
print(train_de_indices[1])
print(len(en_vocab))
print(len(de_vocab))

##note: train here is only 50 example! do not forget to change this
#train_en_indices = train_en_indices[0:50]
#train_de_indices = train_de_indices[0:50]


def create_vocab_dicts(data, prev_vocab):
  vocab = {}
  vocab["CLS"] = 0
  vocab["SEP"] = 1
  vocab["PAD"] = 2
  vocab["UNK"] = 3
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
    

class EncoderAttention(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(EncoderAttention, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
  
  def forward(self, indices):
    embeddings = self.embeddings(indices)
    output, (h_n, c_n) = self.lstm(embeddings)
    return output, h_n, c_n
  
class DecoderAttention(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(DecoderAttention, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    self.linear = nn.Linear(hidden_size*2, vocab_size, bias=True)
    self.attention_train = AttentionTrain(hidden_size) #attention module that we define
    self.attention_eval  = AttentionEval(hidden_size)
    #for test time:
    self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size, bias=True)
    
    self.lstm_cell.weight_ih = self.lstm.weight_ih_l0
    self.lstm_cell.weight_hh = self.lstm.weight_hh_l0
    self.lstm_cell.bias_ih = self.lstm.bias_ih_l0
    self.lstm_cell.bias_hh = self.lstm.bias_hh_l0
    
    
  def forward(self, indices, enc_output, h_0, c_0, use_ground_truth=True):
    use_ground_truth = True    
    
    if self.training == False:
      max_sentence_len = indices.size()[1]
      batch_size = indices.size()[0]
      next_word_idx = torch.zeros([batch_size,1]).long().to(device) # Make the first LSTM input [CLS]
      eos_flags = torch.zeros(batch_size).long().to(device)
      # Squeezing the dim 0, because we are using LSTMCell and we don't have LSTM layers (yet)
      h_n = h_0.squeeze(0)
      c_n = c_0.squeeze(0)
      scores = None
      i = 0      
      while int(torch.sum(eos_flags)) < batch_size:
        # LSTMCell expects and input (batch, input_size), squeeze seq_len
        embeddings = self.embeddings(next_word_idx).squeeze(1)
        h_n, c_n = self.lstm_cell(embeddings, (h_n, c_n))
        # Apply unsqueeze to output so that it matches the dimensions of the training output
        h_n_attn = self.attention_eval(enc_output, h_n)

        
        if scores is None:
          scores = self.linear(h_n_attn)
          next_word_idx = torch.argmax(scores.squeeze(1), dim=1)
        else:
          scores_n = self.linear(h_n_attn)
          next_word_idx = torch.argmax(scores_n.squeeze(1), dim=1)
          scores = torch.cat((scores, scores_n), dim=1)
        
        # Doing greedy search by getting the position of the highest value in the squence
        # next_word_idx = torch.argmax(scores[:, -1, :].squeeze(1), dim=1)
        # Check if there are EOS, so in our case [SEP]. Index is 1
        flags_to_add = (next_word_idx[:] == 1).long()
        eos_flags += flags_to_add
        # Just in case an [SEP] is generated twice
        eos_flags[eos_flags > 1] = 1
        
        # Unsqueezing an extra dimension to match the input for the next iteration
        next_word_idx = next_word_idx.unsqueeze(1)
        if i > max_sentence_len-2:
          #print("Sentence is too long:", i)
          break
        i+=1
      return scores
    
    
    if (self.training == True and use_ground_truth == True): #model is in training mode
      embeddings = self.embeddings(indices)
      dec_output, (h_n, c_n) = self.lstm(embeddings, (h_0, c_0))      
      attn_output = self.attention_train(enc_output, dec_output)
    
      scores = self.linear(attn_output)
      return scores

from torch.utils.data import Dataset, DataLoader

class LangDataset(Dataset):
    """Language dataset dataset."""

    def __init__(self, source_sentences, target_sentences):
        """
        Args:
            source_sentences (torch Tensor): Contains all the sentences
            from the source language represented with indices.
            target_sentences (torch Tensor): Contains all the sentences
            from the target language represented with indices.
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return self.source_sentences[idx], self.target_sentences[idx]

def train(lang_dataset, params, encoder, decoder):
    
    # since the second index corresponds to the PAD token, we just ignore it
    # when computing the loss
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    
    optim_encoder = optim.Adam(encoder.parameters(), lr=params['learning_rate'])
    optim_decoder = optim.Adam(decoder.parameters(), lr=params['learning_rate'])
    
    dataloader = DataLoader(lang_dataset, batch_size=10, shuffle=True, num_workers=0)
    
    ground_truth_prob = params['ground_truth_prob']
    
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        
        ground_truth_prob *= params['prob_decay_rate']
        
        # for each batch, calculate loss and optimize model parameters            
        for (de_indices, en_indices) in dataloader:
          
            x = torch.distributions.uniform.Uniform(0,1).sample()
            
            #h_n, c_n = encoder(de_indices)
            #preds = decoder(en_indices[:,0:-1], h_n, c_n, ground_truth_prob > x)
            
            enc_outputs, h_n, c_n = encoder(de_indices)
            preds = decoder(en_indices[:,0:-1],enc_outputs, h_n, c_n, ground_truth_prob > x)
            
            targets = en_indices[:,1:]
            # preds = preds.view(10, 24196, 815)
            # preds = preds.view(params['batch_size'], params['vocab_size'], -1)
            preds = preds.permute(0, 2, 1)
            
            # print(preds.size()[2])
            targets = targets[:,0:preds.size()[2]]
            loss = criterion(preds, targets)
            
            loss.backward()
            optim_encoder.step()
            optim_encoder.zero_grad()
            optim_decoder.step()
            optim_decoder.zero_grad()
            ep_loss += loss

        with open("loss.txt", "a+") as l:
            l.write("Epoch: " + str(epoch) + " Loss: " + str(ep_loss) + " time: " + str(time.time()-start_time) + "\n")
                
        torch.save(encoder.state_dict(), './models/encoder_' + str(epoch) + '.pt')
        torch.save(decoder.state_dict(), './models/decoder_' + str(epoch) + '.pt')

params = {}
params['vocab_size'] = len(en_vocab)
params['batch_size'] = 150
params['epochs'] = 50
params['learning_rate'] = 1e-5
params['ground_truth_prob'] = 1
params['prob_decay_rate'] = 1
hidden_size = 512

encoder = EncoderAttention(int(torch.max(train_de_indices)) + 1, hidden_size)
decoder = DecoderAttention(int(torch.max(train_en_indices)) + 1, hidden_size)

lang_dataset = LangDataset(train_de_indices.long().to(device), train_en_indices.long().to(device))
print("ready to train")
train(lang_dataset, params, encoder.to(device), decoder.to(device))
