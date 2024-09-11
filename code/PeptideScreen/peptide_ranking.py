import math
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(1234)
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

import torch
import torch.utils.data as Data
import torch.nn as nn
import difflib

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import os
import argparse
import logging
import sys

parser = argparse.ArgumentParser(usage = 'peptide screen')
parser.add_argument('--mtpeptide_file', type = str, help = 'the path of neoantigens prediction file.')
parser.add_argument('--wtpeptide_file', type = str, help = 'the path of the wild-type peptides prediction file.')
parser.add_argument('--output_dir', type = str, help = 'The directory where the output results are stored.')
parser.add_argument('--sample_id', type = str,  help = 'snv or indel')
args = parser.parse_args()


output_dir = args.output_dir
sample_id = args.sample_id
    
if not args.mtpeptide_file:
    sys.exit(0)
if not args.wtpeptide_file:
    sys.exit(0)

if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)


mt_data = pd.read_csv(args.mtpeptide_file, header=0)

wt_data = pd.read_csv(args.wtpeptide_file, header=0)


merged_df = pd.merge(mt_data, wt_data, on=['#Position','HLA_type','Gene','Transcript_name','Mutation','AA_change',
                                         'MT_pep','WT_pep','HLA_sequence','tpm','DriverGene_Lable','allele_frequency',
                                         'CDR3','cloneCount','cloneFraction'
                                        ])


merged_df = merged_df[merged_df.predicted_label_x == 1]
merged_df = merged_df[merged_df.predicted_label_y == 0].reset_index(drop=True)



merged_df = merged_df[merged_df.tpm != 'none'].reset_index(drop=True)
merged_df.drop_duplicates(subset=['Gene','AA_change','HLA_type', 'MT_pep', 'WT_pep','CDR3','DriverGene_Lable','allele_frequency'], keep='first').reset_index(drop=True)



pep_group = merged_df.groupby(['MT_pep','WT_pep','Gene','tpm','DriverGene_Lable','allele_frequency'])


MT_pep_list = []
WT_pep_list = []
gene_list = []
tpm_list = []
DriverGene_list = []
af_list = []
tcr_clone_list = []
tcr_clone_count = []
max_bindingscore = []

for name, group in pep_group:
  
    #print(name, group)    
    MT_pep_list.append(name[0])
    WT_pep_list.append(name[1])
    gene_list.append(name[2])
    tpm_list.append(name[3])
    DriverGene_list.append(name[4])
    af_list.append(name[5])
    
    tcr_clone_list.append(group['cloneFraction'].values.sum())
    tcr_clone_count.append(len(group['cloneFraction'].values))
    max_bindingscore.append(np.max(group['predicted_score_x'].values))
tpm_list = list(map(float, tpm_list)) 

data = pd.DataFrame({'MT_pep': MT_pep_list, 'WT_pep': WT_pep_list, 'Gene':gene_list,'tpm':tpm_list, 'DriverGene_Lable':DriverGene_list,'allele_frequency':af_list,
                     'tcr_clone_sum': tcr_clone_list,'tcr_clone_conut':tcr_clone_count,'max_binding_score':max_bindingscore})
sorted_df = data.sort_values('tcr_clone_conut', ascending=False)
sorted_df = data.sort_values('tpm', ascending=False)
sorted_df.reset_index(drop=True)
sorted_df.to_csv(output_dir + '/'+ sample_id +'_pep.csv')






