#-*- coding:utf-8 -*-
import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size # 指的是 vocabulary size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        # print(src.size()) # size(src) = (seq_len, batch)
        embedded = self.embed(src)
        # print(embedded.size())  # size(embedded) = (seq_len, batch, embed_size)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        # print(outputs.size())
        # size(outputs) = (seq_len, batch, hidden_size * n_layers)
        # size(hidden) = (n_layers, batch, hidden_size)
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        # print(outputs.size())
        # size(outputs) = (seq_len, batch, hidden_size)
        return outputs, hidden



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # size(hidden) = (batch, hidden dim);
        # size(encoder_outputs) = (encoder input seq len, batch, hidden dim)
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        # size(h) = (batch, encoder input seq len, hidden dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        # size(encoder_outputs) = (batch, encoder input seq len, hidden dim)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]




class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        # print("1",input.size(),last_hidden.size(),encoder_outputs.size())
        # size(input) = (batch,);
        # size(last_hidden) = (1, batch, hidden dim);
        # size(encoder_outputs) = (encoder input seq len, batch, hidden dim)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N) = (1, batch, embed dim)
        # print("2",embedded.size())
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # print("3", attn_weights.size())
        # size(attn_weights) = (batch, 1, encoder input seq len)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2) # (1, batch, embed dim + hidden dim)
        # print("4", rnn_input.size())
        output, hidden = self.gru(rnn_input, last_hidden)
        # print("5", output.size(),hidden.size())
        # size(output) = size(hidden) = (1, batch, hidden dim)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        # print("6", output.size(),context.size())
        # size(output) = size(context) = (batch, hidden dim)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        # print("7", output.size(),hidden.size(),attn_weights.size())
        # size(output) = (batch, decoder vocab_size)
        # size(hidden) = (n_layers, batch, hidden dim)
        # size(attn_weights) = (batch, 1, encoder input seq len)
        return output, hidden, attn_weights




class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        # size(src) = (seq_len, batch); size(trg) = (seq_len, batch)
        print("src,trg,size", src.size(),trg.size())

        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            # print(output.size(),hidden.size(),attn_weights.size())
            # size(output) = (batch, decoder vocab_size)
            # size(hidden) = (n_layers, batch, hidden_dim)
            # size(attn_weights) = (batch, encoder_seq_len)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs
