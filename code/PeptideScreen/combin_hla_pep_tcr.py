from __future__ import division
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import difflib

seed = 19961231
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import os
import argparse
import logging
import sys

parser = argparse.ArgumentParser(usage = 'merge tcr-phla data')
parser.add_argument('--peptide_file', type = str, help = 'the path of file contains mutate peptides, wild peptides and HLA_types drived from patient (*.csv).')
parser.add_argument('--TCR_file', type = str, help = 'the path of file contains TCR sequence drived from patient (*.csv).')
parser.add_argument('--output_dir', type = str, help = 'The directory where the output results are stored.')
parser.add_argument('--sample_id', type = str,  help = 'snv or indel')
args = parser.parse_args()
print(args)

output_dir = args.output_dir
sample_id = args.sample_id
    
if not args.peptide_file:
    sys.exit(0)
if not args.TCR_file:
    sys.exit(0)

if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)



tcr_file = args.TCR_file
tcr_data = pd.read_csv(tcr_file,header=0)


tcr_data = tcr_data[['CDR3','cloneCount','cloneFraction']]


pep_file = args.peptide_file
neo_data = pd.read_csv(pep_file,header=0)
    

name = list(neo_data) + list(tcr_data)

data = []
for i in list(neo_data.index):
    for j in list(tcr_data.index):
        temp = list(neo_data.iloc[i])
        temp.extend(tcr_data.iloc[j])
        data.append(temp)

pd_data = pd.DataFrame(data=data, columns=name)


pd_data.to_csv(output_dir + '/'+ sample_id +'_phla_tcr.csv', index=False)












