import torch.utils.data as Data


class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, tcr_inputs, labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.tcr_inputs = tcr_inputs
        self.labels = labels

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.tcr_inputs[idx], self.labels[idx]