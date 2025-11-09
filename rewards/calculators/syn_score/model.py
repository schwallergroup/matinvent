from __future__ import print_function, division

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, atom_fea_len=90, h_fea_len=180, n_h=1):
        super(Net, self).__init__()
        self.cgnf_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.cgnf_to_fc_softplus = nn.Softplus()
        self.final_fea = 0

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])

        self.fc_out = nn.Linear(h_fea_len, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout()

    def forward(self, struc_fea):
        comp_fea = self.cgnf_to_fc(struc_fea)
        comp_fea = self.cgnf_to_fc_softplus(comp_fea)
        comp_fea = self.dropout(comp_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                comp_fea = softplus(fc(comp_fea))

        self.final_fea = comp_fea

        out = self.fc_out(comp_fea)
        out = self.logsoftmax(out)
        return out
