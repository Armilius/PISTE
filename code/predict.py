from __future__ import division
import numpy as np
import pandas as pd
import random
import warnings
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from Model.PISTE import Transformer

import os
import argparse
import sys


parser = argparse.ArgumentParser(usage = 'TCR-ANTIGEN-HLA binding prediction')
parser.add_argument('--input', type = str, help = 'the path to the input data file (*.csv).')
parser.add_argument("--model_name", default='random', type=str, choices=['random', 'unipep', 'reftcr'],
                        help="Choose different trained model by using datasets generated by different negative datasampling.")
parser.add_argument('--threshold', type = float, default = 0.5, help = 'the threshold to define predicted binder, float from 0 - 1, the recommended value is 0.5')
parser.add_argument('--antigen_type', type = str, default = 'MT', help = 'the antigen type, choice["MT","WT"]')
parser.add_argument('--output', type = str, help = 'The directory where the output results are stored(*.csv).')
args = parser.parse_args()



errLogPath = args.output + '/error.log'
if not os.path.exists(args.output): os.makedirs(args.output)
# if args.threshold <= 0 or args.threshold >= 1:
#     log = Logger(errLogPath)
#     log.logger.critical('The threshold invalid, please check whether it ranges from 0-1.')
#     sys.exit(0)
#
# errLogPath = args.output + '/error.log'
# if not args.input:
#     log = Logger(errLogPath)
#     log.logger.critical('The input file is empty.')
#     sys.exit(0)
# if not args.output:
#     log = Logger(errLogPath)
#     log.logger.critical('Please fill the output file directory.')
#     sys.exit(0)
# if not os.path.exists(args.output): os.makedirs(args.output)


random.seed(1234)
warnings.filterwarnings("ignore")

pep_max_len = 11
hla_max_len = 34
tcr_max_len = 30
tgt_len = pep_max_len + hla_max_len + tcr_max_len

batch_size = 1024
# epochs = 60
threshold = 0.5
d_model = 64
dim = 64
d_ff = 512
e_layers = 3
n_heads = 9
sigma = 1
d_layers = 1
interact_layers = 1
window_size = '3'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
vocab = {'C': 1, 'W': 2, 'V': 3, 'A': 4, 'H': 5, 'T': 6, 'E': 7, 'K': 8, 'N': 9, 'P': 10, 'I': 11, 'L': 12, 'S': 13, 'D': 14, 'G': 15, 'Q': 16, 'R': 17, 'Y': 18, 'F': 19, 'M': 20, '-': 0}
vocab_size = len(vocab)
f_mean = lambda l: sum(l) / len(l)



def read_predict_data(predict_data, antigen_type, batch_size):
    
    column_names = predict_data.columns
    if "HLA_sequence" not in column_names:
    
        hla_sequence = pd.read_csv(r'../data/raw_data/common_hla_sequence.csv')
        predict_data = pd.merge(predict_data, hla_sequence, on = 'HLA_type')
        
    pep_inputs, hla_inputs, tcr_inputs = make_data(predict_data, antigen_type)
    loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, tcr_inputs), batch_size, shuffle=False, num_workers=0)
    return predict_data, pep_inputs, hla_inputs, tcr_inputs, loader

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, tcr_inputs):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.tcr_inputs = tcr_inputs

#
    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.tcr_inputs[idx]



class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def make_data(data, type):
    pep_inputs, hla_inputs, tcr_inputs = [], [], []
    num = 0
    
    if type == 'WT':
        for pep, hla, tcr in zip(data.WT_pep, data.HLA_sequence, data.CDR3):
            pep, hla, tcr = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-'), tcr.ljust(tcr_max_len, '-')

            pep_input = [[vocab[n] for n in pep]]
            hla_input = [[vocab[n] for n in hla]]
            tcr_input = [[vocab[n] for n in tcr]]
            pep_inputs.extend(pep_input)
            hla_inputs.extend(hla_input)
            tcr_inputs.extend(tcr_input)
            
            num = num + 1
            
    else:
        for pep, hla, tcr in zip(data.MT_pep, data.HLA_sequence, data.CDR3):
            pep, hla, tcr = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-'), tcr.ljust(tcr_max_len, '-')

            pep_input = [[vocab[n] for n in pep]]
            hla_input = [[vocab[n] for n in hla]]
            tcr_input = [[vocab[n] for n in tcr]]
            pep_inputs.extend(pep_input)
            hla_inputs.extend(hla_input)
            tcr_inputs.extend(tcr_input)
            
            num = num + 1
    

    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(tcr_inputs)


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def eval_step(model, val_loader, threshold = 0.5, use_cuda = False):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.eval()
    torch.manual_seed(19961231)
    torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        y_prob_val_list, dec_attns_val_list = [], []
        for val_pep_inputs, val_hla_inputs, val_tcr_inputs in val_loader:
            val_pep_inputs, val_hla_inputs, val_tcr_inputs = val_pep_inputs.to(device), val_hla_inputs.to(device), val_tcr_inputs.to(device)
            
            val_outputs, _, val_dec_self_attns = model(val_pep_inputs, val_hla_inputs, val_tcr_inputs)
            
            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_prob_val_list.extend(y_prob_val)
            
            dec_attns_val_list.extend(val_dec_self_attns[0][:, :, 15:, :15]) 
                    
        y_pred_val_list = transfer(y_prob_val_list, threshold)
    
    return y_pred_val_list, y_prob_val_list, dec_attns_val_list

antigen_type = args.antigen_type
predict_data = pd.read_csv(args.input)


predict_data, predict_pep_inputs, predict_hla_inputs, predict_tcr_inputs,predict_loader = read_predict_data(predict_data, antigen_type, batch_size)


model = Transformer(device=device,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    e_layers=e_layers,
                    d=dim,
                    n_heads=n_heads,
                    sigma=sigma,
                    window_threshold=window_size,
                    d_ff=d_ff,
                    interact_layers=interact_layers,
                    tgt_len=tgt_len,
                    hla_max_len=hla_max_len,
                    d_layers=d_layers).to(device)
criterion = WeightedFocalLoss(alpha=.75)
optimizer = optim.Adam(model.parameters(), lr=1e-3)




model_dir = r'./checkpoints/'
dir_saver = os.path.join(model_dir, args.model_name)
path_saver = os.path.join(dir_saver, "exp0.pkl")
#print(path_saver)


model.load_state_dict(torch.load(path_saver, map_location='cuda:0'))
model_eval = model.eval()

y_pred, y_prob, attns = eval_step(model_eval, predict_loader, 0.5, use_cuda)

predict_data['predicted_label'], predict_data['predicted_score'] = y_pred, y_prob
predict_data.to_csv(args.output + '/pre_out.csv',index=0)








