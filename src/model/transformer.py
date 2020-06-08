"""
NOT USED
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FFN(nn.Module):
    def __init__(self, model_dim, dff):
        super(FFN, self).__init__()
        self.l1 = nn.Linear(model_dim, dff)
        self.l2 = nn.Linear(dff, model_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        out = self.l2(x)
        return out
    
class MultiHeadAttn(nn.Module):
    def __init__(self, model_dim, q_dim, h):
        super(MultiHeadAttn, self).__init__()
        self.model_dim = model_dim
        self.q_dim = q_dim
        self.h = h
        
        self.wq = nn.Linear(model_dim, q_dim * h)
        self.wk = nn.Linear(model_dim, q_dim * h)
        self.wv = nn.Linear(model_dim, q_dim * h)
        
        self.out = nn.Linear(q_dim * h, model_dim)
        
    def forward(self, q, k, v, mask=None):
        q = self.wq(q).view(q.size(0), -1, self.h, self.q_dim) #(B, T, h, dq)
        k = self.wk(k).view(q.size(0), -1, self.h, self.q_dim)
        v = self.wv(v).view(q.size(0), -1, self.h, self.q_dim)
        
        q = q.transpose(1, 2) #(B, h, T, dq)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        w = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.q_dim)

        if mask is not None:
            w = w.masked_fill(mask, float("-inf"))

        w = F.softmax(w, -1)
        z = torch.matmul(w, v) #bhtt, bhtq -> bhtq
        output = z.transpose(1, 2) #bhtp -> bthq
        output = output.contiguous().view(q.size(0), -1, self.h * self.q_dim)
        output = self.out(output)

        return output
        

class EncoderModule(nn.Module):
    def __init__(self, model_dim, q_dim, h, dff):
        super(EncoderModule, self).__init__()
        self.multihead = MultiHeadAttn(model_dim, q_dim, h)
        self.layernorm1 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, dff)
        self.layernorm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        z = self.layernorm1(self.multihead(x, x, x) + x)
        z = self.layernorm2(self.ffn(z) + z)
        return z

class Encoder(nn.Module):
    def __init__(self, blocks, model_dim, q_dim, h, dff):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([EncoderModule(model_dim, q_dim, h, dff) for _ in range(blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class DecoderModule(nn.Module):
    def __init__(self, model_dim, q_dim, h, dff):
        super(DecoderModule, self).__init__()
        self.multihead1 = MultiHeadAttn(model_dim, q_dim, h)
        self.layernorm1 = nn.LayerNorm(model_dim)
        self.multihead2 = MultiHeadAttn(model_dim, q_dim, h)
        self.layernorm2 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, dff)
        self.layernorm3 = nn.LayerNorm(model_dim)

    def forward(self, q, k, v):
        o = torch.ones(q.shape[1], q.shape[1]).to(device)
        triu = torch.triu(o, 1)
        mask = triu.byte()
        z = self.layernorm1(self.multihead1(q, q, q, mask) + q)
        z = self.layernorm2(self.multihead2(z, k, v) + z)
        z = self.layernorm3(self.ffn(z) + z)
        return z

class Decoder(nn.Module):
    def __init__(self, blocks, model_dim, q_dim, h, dff):
        super(Decoder, self).__init__()
        self.last_out = last_out
        
        self.blocks = nn.ModuleList([DecoderModule(model_dim, q_dim, h, dff) for _ in range(blocks)])
            
    def forward(self, x, k, v):
        for block in self.blocks:
            x = block(x, k, v)
            
        return x

class Transformer(nn.Module):
    def __init__(self, enc_blocks, dec_blocks, model_dim, q_dim, h, dff, seq_len, last_out):
        super(Transformer, self).__init__()
        self.last_out = last_out
            
        self.positional_encoder = PositionalEncoder(model_dim, seq_len)
        self.encoder = Encoder(enc_blocks, model_dim, q_dim, h, dff)
        self.decoder = Decoder(dec_blocks, model_dim, q_dim, h, dff)

    def forward(self, x, decoder_input, max_output=None):
        x = self.positional_encoder(x)
        encoder_output = self.encoder(x)

        if max_output is None:
            decoder_input = self.positional_encoder(decoder_input)
            decoder_output = self.decoder(decoder_input, encoder_output, encoder_output)
            
            return decoder_output
        else:
            while decoder_input.size(1) <= max_output:
                q = self.positional_encoder(decoder_input)
                decoder_output = self.decoder(q, encoder_output, encoder_output)
                decoder_input = torch.cat([decoder_input, decoder_output.max(2)[1][:,-1].unsqueeze(1)], 1)

            return decoder_output

class PositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_seq_len, model_dim)
        for pos in range(max_seq_len):
            for i in range(0, model_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))

                if i != model_dim - 1:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / model_dim)))
                
        pe = pe.unsqueeze(0).detach().to(device)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, blocks, model_dim, q_dim, h, dff, max_seq_len=500):
        super(TransformerEncoder, self).__init__()
            
        self.positional_encoder = PositionalEncoder(model_dim, max_seq_len)
        self.encoder = Encoder(blocks, model_dim, q_dim, h, dff)
        
    def forward(self, x):
        x = self.positional_encoder(x)
        enc_out = self.encoder(x)
        
        return enc_out
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)