import torch
import torch.nn as nn
import random
import warnings
import numpy as np
from .SlidingAttention import Intercoder

random.seed(1234)
warnings.filterwarnings("ignore")


def index2onehot(pep, hla, tcr, device):

    acid = torch.tensor([14, 7, 0]).to(device)
    alkali = torch.tensor([5, 8, 17, 0]).to(device)
    hydrophobic = torch.tensor([19, 4, 12, 20, 11, 2, 10, 3, 0]).to(device)
    hydrogen = torch.tensor([17, 8, 14, 7, 13, 9, 6, 16, 5, 18, 2, 0]).to(device)

    pep_code = torch.zeros_like(pep).type('torch.FloatTensor').to(pep.device).unsqueeze(-1).repeat(1, 1, 4)
    hla_code = torch.zeros_like(hla).type('torch.FloatTensor').to(hla.device).unsqueeze(-1).repeat(1, 1, 4)
    tcr_code = torch.zeros_like(tcr).type('torch.FloatTensor').to(tcr.device).unsqueeze(-1).repeat(1, 1, 4)

    pep_code[:, :, 0][torch.isin(pep, acid)] = 1.
    hla_code[:, :, 0][torch.isin(hla, acid)] = 1.
    tcr_code[:, :, 0][torch.isin(tcr, acid)] = 1.

    pep_code[:, :, 1][torch.isin(pep, alkali)] = 1.
    hla_code[:, :, 1][torch.isin(hla, alkali)] = 1.
    tcr_code[:, :, 1][torch.isin(tcr, alkali)] = 1.

    pep_code[:, :, 2][torch.isin(pep, hydrophobic)] = 1.
    hla_code[:, :, 2][torch.isin(hla, hydrophobic)] = 1.
    tcr_code[:, :, 2][torch.isin(tcr, hydrophobic)] = 1.

    pep_code[:, :, 3][torch.isin(pep, hydrogen)] = 1.
    hla_code[:, :, 3][torch.isin(hla, hydrogen)] = 1.
    tcr_code[:, :, 3][torch.isin(tcr, hydrogen)] = 1.

    return pep_code, hla_code, tcr_code


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.shape[0], seq_q.shape[1]
    batch_size, len_k = seq_k.shape[0], seq_k.shape[1]
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_seq_pad_mask(seq):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len = seq.shape[0], seq.shape[1]
    # eq(zero) is PAD token
    seq_attn_mask = seq.data.eq(0)  # [batch_size, 1, len_k], False is masked
    return seq_attn_mask  # [batch_size, len_q, len_k]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, device, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(EncoderLayer, self).__init__()
        self.cov1 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        self.bn1 = torch.nn.BatchNorm1d(d_model)
        self.relu = torch.nn.ReLU(inplace=True)
        self.cov2 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        self.bn2 = torch.nn.BatchNorm1d(d_model)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        x = self.cov1(enc_inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.cov2(x)
        x = self.bn2(x)

        x = x + enc_inputs
        enc_outputs = self.relu(x)

        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, e_layers):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model) for _ in range(e_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = enc_outputs.transpose(1, 2)
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs = layer(enc_outputs)
        enc_outputs = enc_outputs.transpose(2, 1)
        return enc_outputs


class HLAEncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(HLAEncoderLayer, self).__init__()

        self.convBlock1 = EncoderLayer(d_model=d_model)
        self.convBlock2 = EncoderLayer(d_model=d_model)
        self.convBlock3 = EncoderLayer(d_model=d_model)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        HLA_1 = enc_inputs[:, :, 4:18]
        HLA_2 = torch.cat([enc_inputs[:, :, :4], enc_inputs[:, :, 18:24]], dim=-1)
        HLA_3 = enc_inputs[:, :, 24:]

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        HLA_1_outputs = self.convBlock1(HLA_1)
        HLA_2_outputs = self.convBlock2(HLA_2)
        HLA_3_outputs = self.convBlock3(HLA_3)

        enc_outputs = torch.cat([HLA_2_outputs[:, :, :4], HLA_1_outputs, HLA_2_outputs[:, :, 4:], HLA_3_outputs], dim=-1)
        return enc_outputs


class HLAEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, e_layers):
        super(HLAEncoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([HLAEncoderLayer(d_model) for _ in range(e_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = enc_outputs.transpose(1, 2)
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs = layer(enc_outputs)
        enc_outputs = enc_outputs.transpose(2, 1)
        return enc_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d, n_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d = d
        self.n_heads = n_heads
        self.device = device
        self.W_Q = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d, d_model, bias=False)
    def forward(self, input_Q, input_V, attn):

        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        attn = nn.Softmax(dim=-1)(attn)
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)
        context = torch.matmul(attn, V) + (1 - attn.sum(-1).unsqueeze(-1)) * Q
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d)
        output = self.fc(context)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d, n_heads, device, d_ff):
        super(DecoderLayer, self).__init__()
        self.pep_self_attn = MultiHeadAttention(d_model, d, n_heads, device)
        self.tcr2pep_attn = MultiHeadAttention(d_model, d, n_heads, device)
        self.tcr2pep2hla_attn =MultiHeadAttention(d_model, d, n_heads, device)

        self.pep_pos_ffn = PoswiseFeedForwardNet(device, d_model, d_ff)
        self.tcr2hla_pos_ffn = PoswiseFeedForwardNet(device, d_model, d_ff)

    def forward(self, hla_inputs, pep_inputs, tcr_inputs, pep_attn_mask, tcr_attn_mask): # dec_inputs = enc_outputs
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        pep_outputs, pep_attn = self.pep_self_attn(hla_inputs, pep_inputs, pep_attn_mask)

        tcr2pep_outputs, tcr2pep_attn = self.tcr2pep_attn(pep_inputs, tcr_inputs, tcr_attn_mask[:, :, :11, :])
        tcr2pep2hla_outputs, tcr2pep2hla_attn = self.tcr2pep2hla_attn(hla_inputs, tcr2pep_outputs, pep_attn_mask)

        pep_outputs = self.pep_pos_ffn(pep_outputs)
        tcr2pep2hla_outputs = self.tcr2hla_pos_ffn(tcr2pep2hla_outputs)
        return pep_outputs, tcr2pep2hla_outputs, pep_attn


class Decoder(nn.Module):
    def __init__(self,d_model, d, n_heads, device, d_ff, d_layers, tgt_len):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d, n_heads, device, d_ff) for _ in range(d_layers)])
        self.tgt_len = tgt_len

    def forward(self, hla_inputs, pep_inputs, tcr_inputs, pep_attn, tcr_attn):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        hla_outputs = hla_inputs
        pep_outputs = pep_inputs
        tcr_outputs = tcr_inputs
        pep_attn = pep_attn.transpose(-1, -2)
        tcr_attn = tcr_attn.transpose(-1, -2)
        dec_self_attns = []

        for layer in self.layers:
            pep_outputs, tcr2hla, dec_self_attn = layer(hla_outputs, pep_outputs, tcr_outputs, pep_attn, tcr_attn)
            dec_self_attns.append(dec_self_attn)

        return pep_outputs, tcr2hla, dec_self_attns


class Transformer(nn.Module):
    def __init__(self, device, vocab_size, d_model, e_layers, d, n_heads, sigma, window_threshold, d_ff, interact_layers, tgt_len, hla_max_len, d_layers):
        super(Transformer, self).__init__()
        self.device = device
        self.pep_encoder = Encoder(vocab_size, d_model, e_layers).to(device)
        self.hla_encoder = HLAEncoder(vocab_size, d_model, e_layers).to(device)
        self.tcr_encoder = Encoder(vocab_size, d_model, e_layers).to(device)

        self.binaryInteraction = Intercoder(d_model, d, n_heads, sigma, window_threshold, device, d_ff, interact_layers).to(device)
        self.ternaryInteraction = Intercoder(d_model, d, n_heads, sigma, window_threshold, device, d_ff, interact_layers).to(device)
        self.decoder = Decoder(d_model, d, n_heads, device, d_ff, d_layers, tgt_len).to(device)
        self.tgt_len = tgt_len
        self.projection = nn.Sequential(
            nn.Linear(3 * hla_max_len * d_model, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),

            # output layer
            nn.Linear(64, 2)
        ).to(device)

        self.dropout = nn.Dropout(0.3)

    def forward(self, pep_inputs, hla_inputs, tcr_inputs):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]

        pep_code, hla_code, tcr_code = index2onehot(pep_inputs, hla_inputs, tcr_inputs, self.device)
        binary_mask = torch.matmul(pep_code, hla_code.transpose(-1, -2))
        ternary_mask = torch.matmul(tcr_code, torch.cat([pep_code, hla_code], dim=-2).transpose(-1, -2))
        binary_mask[binary_mask > 0] = 1
        ternary_mask[ternary_mask > 0] = 1
        ################################################################################################################
        pep_pos = (torch.arange(11).type('torch.FloatTensor').to(self.device) + 1.) * (33 / 11.) - 0.5*(33 / 11.)
        tcr_pos = (torch.arange(30).type('torch.FloatTensor').to(self.device) + 1.) * (33 / 30.) - 0.5*(33 / 30.)
        hla_pos = torch.arange(34).type('torch.FloatTensor').to(self.device)

        pep2hla = torch.exp(-torch.square(hla_pos.unsqueeze(1) - pep_pos.unsqueeze(0))/(2*1.5*1.5))
        tcr2hla = torch.exp(-torch.square(hla_pos.unsqueeze(1) - tcr_pos.unsqueeze(0))/(2*0.5*0.5))
        # print(pep2hla)

        pep2hla[pep2hla < 1e-3] = -np.inf
        pep2hla[pep2hla >= 1e-3] = 1.
        tcr2hla[tcr2hla < 1e-3] = -np.inf
        tcr2hla[tcr2hla >= 1e-3] = 1.

        pep2hla = torch.softmax(pep2hla, dim=-1).unsqueeze(0)  # [batch-size, 34, 11]
        tcr2hla = torch.softmax(tcr2hla, dim=-1).unsqueeze(0)  # [batch-size, 34, 30]
        ################################################################################################################

        pepHLA_mask = get_attn_pad_mask(pep_inputs, hla_inputs)
        HLApep_mask = get_attn_pad_mask(hla_inputs, pep_inputs)
        TCRpHLA_mask = get_attn_pad_mask(tcr_inputs, torch.cat([pep_inputs, hla_inputs], dim=-1))
        pHLATCR_mask = get_attn_pad_mask(torch.cat([hla_inputs, pep_inputs], dim=-1), tcr_inputs)

        ################################################################################################################

        pep_matrix = pep_inputs.clone()
        tcr_matrix = tcr_inputs.clone()
        pep_matrix[pep_matrix > 0] = 1
        tcr_matrix[tcr_matrix > 0] = 1
        len_pep = pep_matrix.sum(-1).unsqueeze(-1)
        len_tcr = tcr_matrix.sum(-1).unsqueeze(-1)

        ################################################################################################################

        pep_enc_outputs = self.pep_encoder(pep_inputs)
        hla_enc_outputs = self.hla_encoder(hla_inputs)
        pep_inter_outputs, hla_inter_outputs, binary_pep_attn, binary_hla_attn, pep_attn = self.binaryInteraction(pep_enc_outputs,
                                                                                                        hla_enc_outputs,
                                                                                                        binary_mask,
                                                                                                        pepHLA_mask,
                                                                                                        HLApep_mask,
                                                                                                        len_pep,
                                                                                                        None)
        binary_enc_outputs = torch.cat((pep_enc_outputs + pep_inter_outputs, hla_enc_outputs + hla_inter_outputs), 1)

        tcr_enc_outputs = self.tcr_encoder(tcr_inputs)
        tcr_inter_outputs, binary_inter_outputs, tribble_attn, _, tcr_attn = self.ternaryInteraction(tcr_enc_outputs,
                                                                                           binary_enc_outputs,
                                                                                           ternary_mask,
                                                                                           TCRpHLA_mask,
                                                                                           pHLATCR_mask,
                                                                                           len_tcr,
                                                                                           len_pep)

        ################################################################################################################
        hla_dec_embedding = (binary_enc_outputs + binary_inter_outputs)[:, 11:, :]
        pep_dec_embedding = (binary_enc_outputs + binary_inter_outputs)[:, :11, :]
        tcr_dec_embedding = (tcr_enc_outputs + tcr_inter_outputs)

        pep_outputs, tcr2hla_outputs, dec_self_attns = self.decoder(hla_dec_embedding, pep_dec_embedding, tcr_dec_embedding, pep_attn, tcr_attn)
        pep_embedding = torch.matmul(pep2hla, pep_dec_embedding)
        tcr_embedding = torch.matmul(tcr2hla, tcr_dec_embedding)
        ################################################################################################################

        ternary_enc_outputs = torch.cat((tcr_embedding + tcr2hla_outputs, pep_embedding + pep_outputs, hla_dec_embedding), 1)
        dec_outputs = ternary_enc_outputs.view(ternary_enc_outputs.shape[0], -1)
        dec_logits = self.dropout(self.projection(dec_outputs))

        return dec_logits.view(-1, dec_logits.size(-1)), binary_pep_attn, tribble_attn