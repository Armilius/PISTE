import csv
import numpy as np
import pandas as pd
import random
import collections
from collections import defaultdict
from sklearn.utils import shuffle 

import argparse
import sys


def random_shuffle(pos_file,neg_file,num):
    data0 = []
    data1 = []
    data2 = []
    data = {}
    with open(pos_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for id, row in enumerate(spamreader):
            if id > 0:
                data0.append(row[0])
                data1.append(row[1])
                data2.append(row[2])
                data[row[0]] = [row[1],row[2]]

    n = num
    res = collections.defaultdict(list)
    
    with open(neg_file,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["CDR3","MT_pep","HLA_type","Label","n"])

        for i in range(n):
            d1 = data1.copy()
            d2 = data2.copy()
            for id, j in enumerate(data0):
                pos = data[j]
                
                c1 = random.choice(d1)
                c2 = random.choice(d2)
                
                while c1 == pos[0] and c2 == pos[1]:
                    c1 = random.choice(d1)
                    c2 = random.choice(d2)
                res[i].append([j, c1, c2, '0', i])

            for i in res[i]:
                writer.writerow(i)
    
    neg_data = pd.read_csv(neg_file)
    del neg_data['n']
    neg_data2 = neg_data.drop_duplicates(subset=['CDR3', 'MT_pep', 'HLA_type']).reset_index(drop = True)  
    neg_data2.to_csv(neg_file,index=None) 


def unified_pep(pos_file,neg_file,num):
    pos_data = pd.read_csv(pos_file, header=0)
    peps = pos_data['MT_pep'].values 
    ps = set(peps)        
    tcrs = pos_data['CDR3'].values 
    ts = set(tcrs)
    hlas = pos_data['HLA_type'].values 
    hs = set(hlas)

    p2t_pos = defaultdict(set)
    for i in range(len(tcrs)):
        p2t_pos[peps[i]].add(tcrs[i])
        
    p2t_neg = dict()
    for i in p2t_pos.keys():
        p2t_neg[i] = list(ts - p2t_pos[i])
               
    p2h_pos = defaultdict(set)
    for i in range(len(hlas)):
        p2h_pos[peps[i]].add(hlas[i])
        
    p2h_neg = dict()
    for i in p2h_pos.keys():
        p2h_neg[i] = list(hs - p2h_pos[i])

    neg_tcrs, neg_peps, neg_hlas = sample_neg_tcr(p2t_neg,peps,hlas,p2h_neg,num)
    neg_data = {'CDR3': neg_tcrs,
            'MT_pep':neg_peps,
            'HLA_type':neg_hlas   
    }
    neg_df = pd.DataFrame(neg_data)
    neg_df['Label'] = 0
    neg_data2 = neg_df.drop_duplicates(subset=['CDR3','MT_pep','HLA_type','Label']).reset_index(drop = True)
    neg_data2.to_csv(neg_file,index=None)


def sample_neg_tcr(p2t_neg,peps,hlas,p2h_neg,num):
    neg_peps = []
    neg_hlas = []
    neg_tcrs = [] 
    n = int(num/2)   
    i = 0
    for e in peps:                       
        neg_peps.extend([e]*n)                                                
        neg_tcrs.extend(np.random.choice(p2t_neg[e],n,replace=False))
        neg_hlas.extend(np.random.choice(p2h_neg[e],n,replace=False)) 
    
    for e in peps:                       
        
        hla = hlas[i]
        neg_peps.extend([e]*n)                                                
        neg_tcrs.extend(np.random.choice(p2t_neg[e],n,replace=False))
        neg_hlas.extend([hla]*n)
        i += 1
    
    return neg_tcrs, neg_peps, neg_hlas


def reference_TCR(pos_file,neg_file,ref_file,num):
    pos_data = pd.read_csv(pos_file, header=0) 
    reference_tcr = pd.read_csv(ref_file)['CDR3']
    peps = pos_data['MT_pep'].values #store epitopes
    es = set(peps)        
    tcrs = pos_data['CDR3'].values #store tcrs
    ts = set(tcrs)
    hlas = pos_data['HLA_type'].values #store epitopes
    hs = set(hlas) 
    neg_tcrs, neg_peps, neg_hlas = sample_neg_reftcr(peps,hlas,reference_tcr,num)
    neg_data = {'CDR3': neg_tcrs,
            'MT_pep':neg_peps,
            'HLA_type':neg_hlas   
    }
    neg_df = pd.DataFrame(neg_data)
    neg_df['Label'] = 0
    neg_data2 = neg_df.drop_duplicates(subset=['CDR3','MT_pep','HLA_type','Label']).reset_index(drop = True)
    neg_data2.to_csv(neg_file,index=None)


def sample_neg_reftcr(peps,hlas,reference_tcr,num):
    neg_peps = []
    neg_hlas = []
    neg_tcrs = [] 
    n = num
    i=0
    for e in peps:                              
        hla = hlas[i]
        neg_peps.extend([e]*n)                                                
        neg_tcrs.extend(np.random.choice(reference_tcr,n,replace=False)) 
        neg_hlas.extend([hla]*n)
        i += 1     
    return neg_tcrs, neg_peps, neg_hlas  


def preprocess(pos_file,neg_file): 

    hla_sequence = pd.read_csv(r'../data/raw_data/common_hla_sequence.csv')
    pos_data = pd.read_csv(pos_file, header=0)
    neg_data = pd.read_csv(neg_file, header=0)

    final_pos_neg = pd.concat([pos_data,neg_data],axis=0).reset_index(drop = True)
    final_pos_neg=final_pos_neg.drop_duplicates(subset=['CDR3','MT_pep','HLA_type']).reset_index(drop = True)

    data = pd.merge(final_pos_neg, hla_sequence, on = 'HLA_type')
    data2 = data[data['HLA_type'].str.contains('HLA')]
    data2['peplen'] = data2['MT_pep'].str.len()
    data3 = data2[data2.peplen <= 11].reset_index(drop = True)
    data3 = data3[data3.peplen >= 8].reset_index(drop = True)
    data3=data3.dropna(axis=0).reset_index(drop = True)

    data3["contain_X"]=['X' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_X==True].reset_index(drop = True)
    del data3['contain_X']
    data3["contain_U"]=['U' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_U==True].reset_index(drop = True)
    del data3['contain_U']
    data3["contain_O"]=['O' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_O==True].reset_index(drop = True)
    del data3['contain_O']
    data3["contain_B"]=['B' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_B==True].reset_index(drop = True)
    del data3['contain_B']

    data3["contain_X"]=['X' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_X==True].reset_index(drop = True)
    del data3['contain_X']
    data3["contain_U"]=['U' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_U==True].reset_index(drop = True)
    del data3['contain_U']
    data3["contain_O"]=['O' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_O==True].reset_index(drop = True)
    del data3['contain_O']
    data3["contain_B"]=['B' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3=data3[data3.contain_B==True].reset_index(drop = True)
    del data3['contain_B']
    final_data = data3
    return final_data

def drop_duplicate(df1: pd.DataFrame, df2: pd.DataFrame, name_df_out: str) -> pd.DataFrame:
    df1["mix"] = df1["MT_pep"] + ' ' + df1["HLA_type"]
    df2["mix"] = df2["MT_pep"] + ' ' + df2["HLA_type"]
    d_dup = df2["mix"].isin(df1["mix"])
    df2_drop = df2[~d_dup].drop("mix", axis=1)
    df2_drop.to_csv(name_df_out, index=False)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(usage = 'model evalution')
    parser.add_argument('--pos', type = str, help = 'the path to the positive data file (*.csv).')
    parser.add_argument('--neg', type = str, help = 'the path to the negative data file (*.csv).')
    parser.add_argument('--sampling', type = str, help = 'the negative sampling method.')
    parser.add_argument('--ref_tcr', type = str, help = 'the path to the referene TCRs data file (*.csv).')
    parser.add_argument('--file_type', type = str, help = 'training or test.')
    parser.add_argument('--neg_ratio', default = 10, type = int, help = 'Ratio of negative sampling methods.')
    args = parser.parse_args()

    if not args.pos:
        sys.exit(0)
    if not args.neg:
        sys.exit(0)
    if not args.sampling:
        sys.exit(0)
    if not args.file_type:
        sys.exit(0)

    pos_file = args.pos
    neg_file = args.neg
    sampling = args.sampling
    file_type = args.file_type
    num = args.neg_ratio

    if sampling == 'shuffle':
        random_shuffle(pos_file,neg_file,num)
        data = preprocess(pos_file,neg_file)
        if  file_type == 'training':
            data = shuffle(data) 
            data.to_csv(r'../data/random/train_val.csv',index=None)
            train = data.sample(frac=0.8,random_state=0,axis=0)
            val = data[~data.index.isin(train.index)]
            train.to_csv(r'../data/random/train_data.csv',index=None)
            val.to_csv(r'../data/random/val_data.csv',index=None)
        else:
            data.to_csv(rf'../data/random/{file_type}_data_all.csv',index=None)
            drop_duplicate(
                pd.read_csv("../data/random/train_val.csv"),
                pd.read_csv(rf'../data/random/{file_type}_data_all.csv'),
                rf'../data/random/{file_type}_data.csv',
                )
    
        
    if sampling == 'unipep':
        unified_pep(pos_file,neg_file,num)
        data = preprocess(pos_file,neg_file)
        if  file_type == 'training':
            data = shuffle(data) 
            data.to_csv(r'../data/unipep/train_val.csv',index=None)
            train = data.sample(frac=0.8,random_state=0,axis=0)
            val = data[~data.index.isin(train.index)]
            train.to_csv(r'../data/unipep/train_data.csv',index=None)
            val.to_csv(r'../data/unipep/val_data.csv',index=None)
        else:
            data.to_csv(rf'../data/unipep/{file_type}_data_all.csv',index=None)
            drop_duplicate(
                pd.read_csv("../data/unipep/train_val.csv"),
                pd.read_csv(rf'../data/unipep/{file_type}_data_all.csv'),
                rf'../data/unipep/{file_type}_data.csv',
                )
            

    if sampling == 'reftcr':   
        reference_TCR(pos_file,neg_file,args.ref_tcr,num)
        data = preprocess(pos_file,neg_file)
        if  file_type == 'training':
            data = shuffle(data) 
            data.to_csv(r'../data/reftcr/train_val.csv',index=None)
            train = data.sample(frac=0.8,random_state=0,axis=0)
            val = data[~data.index.isin(train.index)]
            train.to_csv(r'../data/reftcr/train_data.csv',index=None)
            val.to_csv(r'../data/reftcr/val_data.csv',index=None)
        else:
            data.to_csv(rf'../data/reftcr/{file_type}_data_all.csv',index=None)
            drop_duplicate(
                pd.read_csv("../data/reftcr/train_val.csv"),
                pd.read_csv(rf'../data/reftcr/{file_type}_data_all.csv'),
                rf'../data/reftcr/{file_type}_data.csv',
                )
            
    










