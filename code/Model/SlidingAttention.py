import torch
import torch.nn as nn
import math
import random
import warnings
import numpy as np
from .SubLayer import PoswiseFeedForwardNet

random.seed(1234)
warnings.filterwarnings("ignore")

class DualInterAttention(nn.Module):
    def __init__(self, d_model, d, n_heads, sigma, window_threshold, device):
        super(DualInterAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = d
        self.sigma = sigma
        self.d_model = d_model
        if window_threshold == "3":
            self.window_T = 0.01
        elif window_threshold == "2":
            self.window_T = math.exp(-4/2)
        elif window_threshold == '4':
            self.window_T = math.exp(-16/2)
        else:
            self.window_T = math.exp(-1/2)
        self.device = device
        self.W_Q = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d, d_model, bias=False)

        self.W_Q_v = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_K_v = nn.Linear(d_model, d * n_heads, bias=False)
        self.W_V_v = nn.Linear(d_model, d * n_heads, bias=False)
        self.fc_v = nn.Linear(n_heads * d, d_model, bias=False)

    def initial_position(self, name, device, len_q, len_k):
        len_q = len_q.unsqueeze(1) - 1
        if name == "HLA_pep":
            k_pos = torch.arange(34).type('torch.FloatTensor').unsqueeze(0).unsqueeze(0).to(device)
            q_pos_1 = 13. * (torch.arange(11).type('torch.FloatTensor').unsqueeze(0).unsqueeze(0).to(device) / len_q)  # 夹子1，长度为14
            q_pos_1[q_pos_1 > 13] = 1e4
            q_pos_2 = 9. * (torch.arange(11).type('torch.FloatTensor').unsqueeze(0).unsqueeze(0).to(device) / len_q) + 14.  # 底座，长度为10
            q_pos_2[q_pos_2 > 23] = 1e4
            q_pos_3 = 9 - 9. * (torch.arange(11).type('torch.FloatTensor').unsqueeze(0).unsqueeze(0).to(device) / len_q) + 24.  # 夹子2， 长度为10
            q_pos_3[q_pos_3 < 24] = 1e4
            q_pos = [q_pos_1, q_pos_2, q_pos_3]
        elif name == "HLApep_TCR":
            B = len_k.shape[0]
            len_k = len_k.unsqueeze(1) - 1
            k_pos_1 = torch.arange(11).type('torch.FloatTensor').unsqueeze(0).unsqueeze(0).expand(B, 1, 11).to(device)
            k_pos_1[k_pos_1 > len_k] = 1e4
            k_pos_2 = torch.arange(34).type('torch.FloatTensor').unsqueeze(0).unsqueeze(0).expand(B, 1, 34).to(device)
            k_pos = torch.cat([k_pos_1, k_pos_2], dim=-1)

            q_pos_1 = (torch.arange(30).type('torch.FloatTensor').unsqueeze(0).unsqueeze(0).to(device) + 1.) / ((len_q + 1.) * (22. / 30.))
            q_pos_1[q_pos_1 > 1.] = -1
            q_pos_1[q_pos_1 < 8./22.] = -1
            q_pos_1[q_pos_1 > 0] = 7. / 10
            q_pos_1 = q_pos_1 * len_k
            q_pos_1[q_pos_1 < 0] = -1e4

            q_pos = [q_pos_1]
        else:
            q_pos = None
            k_pos = None
            assert (q_pos != None)
        return q_pos, k_pos

    def generate_amino_matrix(self, Q, K, attn_mask):

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dim)
        full_scores = scores.clone()
        scores.masked_fill_(attn_mask, -np.inf)

        return scores, full_scores

    def generate_space_matrix(self, name, device, key_pos, query_positions, transpose=False):
        spaceMask = None
        if name == "HLA_pep":
            '''
                Query: Peptide
                Key: HLA
            '''
            key = key_pos.unsqueeze(-2)
            q_1 = query_positions[0].unsqueeze(-1)
            q_2 = query_positions[1].unsqueeze(-1)
            q_3 = query_positions[2].unsqueeze(-1)
            k_q_1 = torch.exp(-torch.square(q_1 - key[:, :, :, :14]) / (2 * self.sigma * self.sigma))
            k_q_2 = torch.exp(-torch.square(q_2 - key[:, :, :, 14:24]) / (2 * self.sigma * self.sigma))
            k_q_3 = torch.exp(-torch.square(q_3 - key[:, :, :, 24:34]) / (2 * self.sigma * self.sigma))
            spaceMask = torch.cat([k_q_2[:, :, :, :4], k_q_1, k_q_2[:, :, :, 4:10], k_q_3], dim=-1)

        elif name == "HLApep_TCR":
            '''
                Query: TCR
                Key: HLApep
            '''
            key = key_pos.unsqueeze(-2)
            q_1 = query_positions[0].unsqueeze(-1)
            k_q_1 = torch.exp(-torch.square(q_1 - key[:, :, :, :11]) / (2 * self.sigma * self.sigma))
            B, H, _, _ = k_q_1.shape
            k_q_2 = torch.zeros([30, 14]).type('torch.FloatTensor').to(device).unsqueeze(0).unsqueeze(0).expand(B, H, 30, 14)
            k_q_3 = torch.zeros([30, 10]).type('torch.FloatTensor').to(device).unsqueeze(0).unsqueeze(0).expand(B, H, 30, 10)
            k_q_4 = torch.zeros([30, 10]).type('torch.FloatTensor').to(device).unsqueeze(0).unsqueeze(0).expand(B, H, 30, 10)
            spaceMask = torch.cat([k_q_1, k_q_3[:, :, :, :4], k_q_2, k_q_3[:, :, :, 4:10], k_q_4], dim=-1)

        if transpose:
            return spaceMask.transpose(-1, -2)
        else:
            return spaceMask

    def update_pos(self, amino_att, space_att, query_positions, key_pos, name):
        mask = torch.ones_like(space_att).type('torch.FloatTensor').to(amino_att.device)
        mask[space_att >= self.window_T] = 0.
        # mask[space_att >= 0.01] = 0.
        # mask[space_att >= math.exp(-2)] = 0.
        # mask[space_att >= 0.0003] = 1.
        mask = mask.bool()

        if name == "HLA_pep":
            full_att_1 = torch.softmax(amino_att[:, :, :, 4:18].clone().masked_fill_(mask[:, :, :, 4:18], -np.inf), dim=-1)
            full_att_1 = torch.where(torch.isnan(full_att_1), torch.full_like(full_att_1, 0), full_att_1)
            full_att_2 = torch.softmax(torch.cat([amino_att[:, :, :, :4].clone(), amino_att[:, :, :, 18:24]], dim=-1).masked_fill_(torch.cat([mask[:, :, :, :4], mask[:, :, :, 18:24]], dim=-1), -np.inf), dim=-1)
            full_att_2 = torch.where(torch.isnan(full_att_2), torch.full_like(full_att_2, 0), full_att_2)
            full_att_3 = torch.softmax(amino_att[:, :, :, 24:34].clone().masked_fill_(mask[:, :, :, 24:34], -np.inf), dim=-1)
            full_att_3 = torch.where(torch.isnan(full_att_3), torch.full_like(full_att_3, 0), full_att_3)
            query_positions[0] = torch.matmul(full_att_1, key_pos[:, :, :14].unsqueeze(-1)).squeeze(-1) + (1 - full_att_1.sum(-1)) * query_positions[0]
            query_positions[1] = torch.matmul(full_att_2, key_pos[:, :, 14:24].unsqueeze(-1)).squeeze(-1) + (1 - full_att_2.sum(-1)) * query_positions[1]
            query_positions[2] = torch.matmul(full_att_3, key_pos[:, :, 24:34].unsqueeze(-1)).squeeze(-1) + (1 - full_att_3.sum(-1)) * query_positions[2]
        elif name == "HLApep_TCR":
            full_att_1 = torch.softmax(amino_att[:, :, :, :11].clone().masked_fill_(mask[:, :, :, :11], -np.inf), dim=-1)
            full_att_1 = torch.where(torch.isnan(full_att_1), torch.full_like(full_att_1, 0), full_att_1)
            query_positions[0] = torch.matmul(full_att_1, key_pos[:, :, :11].unsqueeze(-1)).squeeze(-1) + (1 - full_att_1.sum(-1)) * query_positions[0]
        return query_positions

    def forward(self, query, key, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k):

        residual_q = query
        residual_k = key
        B, L_Q, _ = query.shape
        B, L_K, _ = key.shape

        if (L_Q + L_K) == 45:
            name = "HLA_pep"
        elif (L_Q + L_K) == 75:
            name = "HLApep_TCR"
        else:
            name = None

        Q = self.W_Q(query).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        K = self.W_K(key).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        V = self.W_V(key).view(B, -1, self.n_heads, self.dim).transpose(1, 2)

        Q_v = self.W_Q_v(key).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        K_v = self.W_K_v(query).view(B, -1, self.n_heads, self.dim).transpose(1, 2)
        V_v = self.W_V_v(query).view(B, -1, self.n_heads, self.dim).transpose(1, 2)

        context = Q
        context_v = Q_v
        query_positions, key_pos = self.initial_position(name, Q.device, len_q, len_k)
        space_attn = self.generate_space_matrix(name, Q.device, key_pos, query_positions)

        amino_attn, full_amino_attn = self.generate_amino_matrix(context, K, pep_attn_mask)
        amino_attn_v, _ = self.generate_amino_matrix(context_v, K_v, hla_attn_mask)
        full_amino_attn.masked_fill_(amino_mask.unsqueeze(1) < 1.0, -np.inf)
        amino_attn.masked_fill_(amino_mask.unsqueeze(1) < 1.0, -np.inf)
        for i in range(2):
            query_positions = self.update_pos(amino_attn, space_attn, query_positions, key_pos, name)
            space_attn = self.generate_space_matrix(name, Q.device, key_pos, query_positions)

        attn = torch.softmax((amino_attn + space_attn).masked_fill_(amino_mask.unsqueeze(1) < 1.0, -np.inf), dim=-1)
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)
        context = torch.matmul(attn, V) + (1 - attn.sum(-1).unsqueeze(-1)) * context

        attn_v = torch.softmax((amino_attn_v + space_attn.transpose(-1, -2)).masked_fill_(amino_mask.unsqueeze(1).transpose(-1, -2) < 1.0, -np.inf), dim=-1)
        attn_v = torch.where(torch.isnan(attn_v), torch.full_like(attn_v, 0), attn_v)
        context_v = torch.matmul(attn_v, V_v) + (1 - attn_v.sum(-1).unsqueeze(-1)) * context_v

        context = context * torch.sum(attn, dim=-1).unsqueeze(-1)
        context_v = context_v * torch.sum(attn_v, dim=-1).unsqueeze(-1)
        context = context.transpose(1, 2).reshape(B, -1, self.n_heads * self.dim)
        context_v = context_v.transpose(1, 2).reshape(B, -1, self.n_heads * self.dim)
        output = self.fc(context)
        output_v = self.fc_v(context_v)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual_q), nn.LayerNorm(self.d_model).to(self.device)(
            output_v + residual_k), attn, attn_v, (full_amino_attn + space_attn).masked_fill_(amino_mask.unsqueeze(1) < 1.0, -np.inf)


class IntercoderLayer(nn.Module):
    def __init__(self, d_model, d, n_heads, sigma, window_threshold, device, d_ff):
        super(IntercoderLayer, self).__init__()
        self.dual_att = DualInterAttention(d_model=d_model, d=d, n_heads=n_heads, sigma=sigma, window_threshold=window_threshold, device=device)
        self.pep_ffn = PoswiseFeedForwardNet(device=device, d_model=d_model, d_ff=d_ff)
        self.hla_ffn = PoswiseFeedForwardNet(device=device, d_model=d_model, d_ff=d_ff)

    def forward(self, pep, hla, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k):
        pep_outputs, hla_outputs, pep_attn, hla_attn, attn = self.dual_att(pep, hla, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k)
        pep_outputs = self.pep_ffn(pep_outputs)
        hla_outputs = self.hla_ffn(hla_outputs)

        return pep_outputs, hla_outputs, pep_attn, hla_attn, attn


class Intercoder(nn.Module):
    def __init__(self, d_model, d, n_heads, sigma, window_threshold, device, d_ff, interact_layers):
        super(Intercoder, self).__init__()
        self.layers = nn.ModuleList([IntercoderLayer(d_model=d_model, d=d, n_heads=n_heads, sigma=sigma, window_threshold=window_threshold, device=device, d_ff=d_ff) for _ in range(interact_layers)])

    def forward(self, pep, hla, amino_mask, pep_attn_mask, hla_attn_mask, len_q, len_k):
        pep_attns, hla_attns = [], []

        for layer in self.layers:
            pep, hla, pep_attn, hla_attn, attn = layer(pep, hla, pep_attn_mask, hla_attn_mask, amino_mask, len_q, len_k)
            pep_attns.append(pep_attn)
            hla_attns.append(hla_attn)

        return pep, hla, pep_attns, hla_attns, attn