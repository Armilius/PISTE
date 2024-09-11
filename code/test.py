from __future__ import division
import numpy as np
import pandas as pd
import random
import warnings
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from Model.PISTE import Transformer
from Dataset.dataloader import MyDataSet
import argparse
import os

random.seed(1234)
warnings.filterwarnings("ignore")

vocab = {'C': 1, 'W': 2, 'V': 3, 'A': 4, 'H': 5, 'T': 6, 'E': 7, 'K': 8, 'N': 9, 'P': 10, 'I': 11, 'L': 12, 'S': 13, 'D': 14, 'G': 15, 'Q': 16, 'R': 17, 'Y': 18, 'F': 19, 'M': 20, '-': 0}
vocab_size = len(vocab)
f_mean = lambda l: sum(l) / len(l)


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def make_data(data, pep_max_len, hla_max_len, tcr_max_len):
    pep_inputs, hla_inputs, tcr_inputs, labels = [], [], [], []
    num = 0
    for pep, hla, tcr, label in zip(data.MT_pep, data.HLA_sequence, data.CDR3, data.Label):
        pep, hla, tcr = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-'), tcr.ljust(tcr_max_len, '-')

        pep_input = [[vocab[n] for n in pep]]
        hla_input = [[vocab[n] for n in hla]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        tcr_inputs.extend(tcr_input)
        labels.append(label)
        num = num + 1

    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(tcr_inputs), torch.LongTensor(labels)


def data_with_loader(type_='test', batch_size=1024, pep_max_len=11,
                     hla_max_len=34, tcr_max_len=30, datapath='../data/random'):

    data = pd.read_csv(os.path.join(datapath, '{}_data.csv'.format(type_)), index_col=False)

    pep_inputs, hla_inputs, tcr_inputs, labels = make_data(data, pep_max_len, hla_max_len, tcr_max_len)
    loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, tcr_inputs, labels), batch_size, shuffle=False, num_workers=0)

    return data, pep_inputs, hla_inputs, tcr_inputs, labels, loader


def calculate_ppvn(y_true, y_scores, n):

    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    sorted_y_true = [y_true[i] for i in sorted_indices]
    ppvn = sum(sorted_y_true[:n])/n
    return ppvn

def performances(y_true, y_pred, y_prob, data, print_=True):


    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
    try:
        mcc = ((tp * tn) - (fn * fp)) / np.sqrt(np.float((tp + fn) * (tn + fp) * (tp + fp) * (tn + fn)))

    except:
        print('MCC Error: ', (tp + fn) * (tn + fp) * (tp + fp) * (tn + fn))
        mcc = np.nan
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    try:
        recall = tp / (tp + fn)
    except:
        recall = np.nan

    try:
        precision = tp / (tp + fp)
    except:
        precision = np.nan

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = np.nan

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    n = Counter(y_true)[1]

    ppvn = calculate_ppvn(y_true, y_prob, n)


    if print_:
        print('auc={:.4f}|aupr={:.4f}|ppvn={:.4f}'.format(roc_auc, aupr, ppvn))

    return (roc_auc, aupr, ppvn)


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def performances_to_pd(performances_dict):
    metrics_name = ['roc_auc', 'aupr', 'ppvn']
    performances_pd = pd.DataFrame(performances_dict, columns=metrics_name)
    return performances_pd


def eval_step(model, val_loader, fold, device, threshold, data):

    model.eval()
    torch.manual_seed(20220909)
    torch.cuda.manual_seed(20220909)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_hla_inputs, val_tcr_inputs, val_labels in tqdm(val_loader):
            val_pep_inputs, val_hla_inputs, val_tcr_inputs, val_labels = val_pep_inputs.to(device), val_hla_inputs.to(
                device), val_tcr_inputs.to(device), val_labels.to(device)
            val_outputs, _, val_dec_self_attns = model(val_pep_inputs, val_hla_inputs, val_tcr_inputs)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)

        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

        print('Exp-{} ****Test: Loss = {:.6f}'.format(fold, f_mean(loss_val_list)))
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, data, print_=True)
    return ys_val, loss_val_list, metrics_val


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='PISTE', description="Test PISTE model")

    parser.add_argument("--pep_max_len", default=11, type=int, help="The max length of Pep sequence.")
    parser.add_argument("--hla_max_len", default=34, type=int, help="The max length of HLA sequence.")
    parser.add_argument("--tcr_max_len", default=30, type=int, help="The max length of TCR sequence.")
    parser.add_argument("--batch_size", default=1024, type=int, help="The number of batch size.")
    parser.add_argument("--threshold", default=0.5, help="The threshold during evaluation.")
    parser.add_argument("--device", default="cuda", help="Set the device number.")
    parser.add_argument("--alpha", default=0.75, help="Parameter of focal loss.")
    parser.add_argument("--gamma", default=2, help="Parameter of focal loss.")
    parser.add_argument("--lr", default=1e-3, help="Set the learning rate.")
    parser.add_argument("--model_dir", default="./checkpoints/", type=str, help="The path to save the model.")
    parser.add_argument("--d_model", default=64 , type=int, help="The dimension of embedding space.")
    parser.add_argument("--e_layers", default=3, type=int, help="The number of encoder layers.")
    parser.add_argument("--dim", default=64, type=int, help="The dimension of sliding attention.")
    parser.add_argument("--n_heads", default=9, type=int, help="The number of heads.")
    parser.add_argument("--sigma", default=1., help="The bandwidth of space matrix.")
    parser.add_argument("--window_size", default="default", type=str, choices=['1', '2', '3', '4'],
                        help="The parameter to control the window size during the sliding process.")
    parser.add_argument("--d_ff", default=512, type=int, help="The dimension of FeedForward layer.")
    parser.add_argument("--interact_layers", default=1, type=int, help="The number of sliding attention layers.")
    parser.add_argument("--d_layers", default=1, type=int, help="The number of decoder layers.")
    parser.add_argument("--exp", default=1, type=int, help="The number of experiments.")
    parser.add_argument("--data_dir", default='../data/', type=str, help="The root path of data files.")
    parser.add_argument("--data_name", default='unipep', type=str, choices=['random', 'unipep', 'reftcr'],
                        help="Choose different datasets generated by different negative datasampling.")
    parser.add_argument("--result_dir", default='./metrics/', type=str, help="The root path of matrics file.")

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        args.device = "cpu"
    if args.window_size == "default":
        if args.data_name == 'random':
            args.window_size = '3'
        elif args.data_name == 'unipep':
            args.window_size = '1'
        elif args.data_name == 'reftcr':
            args.window_size = '2'

    device = torch.device(args.device)
    tgt_len = args.pep_max_len + args.hla_max_len + args.tcr_max_len
    dir_saver = os.path.join(args.model_dir, args.data_name)
    datapath = os.path.join(args.data_dir, args.data_name)
    result_dir = os.path.join(args.result_dir, args.data_name)

    test_data, test_pep_inputs, test_hla_inputs, test_tcr_inputs, test_labels, test_loader = data_with_loader(
        type_='test', batch_size=args.batch_size, pep_max_len=args.pep_max_len,
        hla_max_len=args.hla_max_len, tcr_max_len=args.tcr_max_len, datapath=datapath)
    dbpepneo_data, dbpepneo_pep_inputs, dbpepneo_hla_inputs, dbpepneo_tcr_inputs, dbpepneo_labels, dbpepneo_loader = \
        data_with_loader(type_='dbpepneo', batch_size=args.batch_size, pep_max_len=args.pep_max_len,
                         hla_max_len=args.hla_max_len, tcr_max_len=args.tcr_max_len, datapath=datapath)
    test_fold_metrics_list, dbpepneo_fold_metrics_list = [], []

    for exp in range(0, args.exp):
        print('=====Exp-{}====='.format(exp))
        print('-----Generate data loader-----')
        print('Exp-{} Label info: Test = {} | dbpepneo = {}'.format(exp, Counter(test_data.Label),
                                                                 Counter(dbpepneo_data.Label)))
        print('-----Compile model-----')
        model = Transformer(device=device,
                            vocab_size=vocab_size,
                            d_model=args.d_model,
                            e_layers=args.e_layers,
                            d=args.dim,
                            n_heads=args.n_heads,
                            sigma=args.sigma,
                            window_threshold=args.window_size,
                            d_ff=args.d_ff,
                            interact_layers=args.interact_layers,
                            tgt_len=tgt_len,
                            hla_max_len=args.hla_max_len,
                            d_layers=args.d_layers).to(device)
        criterion = WeightedFocalLoss(alpha=args.alpha, gamma=args.gamma)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        path_saver = os.path.join(dir_saver, "exp{}.pkl".format(exp))
        print('dir_saver: ', dir_saver)
        print('path_saver: ', path_saver)

        metric_best, ep_best = 0, -1
        time_train = 0

        print('-----Evaluate Results-----')

        print('*****Path saver: ', path_saver)
        model.load_state_dict(torch.load(path_saver, map_location='cuda:0'))
        model_eval = model.eval()

        ys_res_test, loss_res_test_list, metrics_res_test = eval_step(model_eval, test_loader, exp, device,
                                                                      args.threshold, test_data)
        test_fold_metrics_list.append(metrics_res_test)

        ys_res_dbpepneo, loss_res_dbpepneo_list, metrics_res_dbpepneo = eval_step(model_eval, dbpepneo_loader, exp,
                                                                                  device, args.threshold, dbpepneo_data)
        dbpepneo_fold_metrics_list.append(metrics_res_dbpepneo)

        print("Total training time: {:6.2f} sec".format(time_train))

    print('****Test-1 set:')
    d3 = performances_to_pd(test_fold_metrics_list)
    print(d3)
    d3.to_csv(os.path.join(result_dir, 'test-1.csv'))

    print('****Test-2 set:')
    d4 = performances_to_pd(dbpepneo_fold_metrics_list)
    print(d4)
    d4.to_csv(os.path.join(result_dir, 'test-2.csv'))

