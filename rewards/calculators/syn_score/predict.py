import argparse
import os
import json

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from rewards.calculators.syn_score import EMB_PATH, MODEL_PATH
from rewards.calculators.syn_score.data import collate_pool
from rewards.calculators.syn_score.model import Net


def get_dataset(struc_list, emb_path):
    with open(emb_path) as f:
        emb_dict = json.load(f)
    
    comp = []
    comp_emb = []
    for struc in struc_list:
        _emb = np.zeros(90)
        _num = 0
        ele_dict = struc.composition.reduced_composition.get_el_amt_dict()
        for element, number in ele_dict.items():
            _num += number
            _emb += np.array(emb_dict[element]) * number
        comp_emb.append(_emb / _num)
        comp.append(struc.composition.reduced_formula)
    
    data = []
    for _comp, _emb in zip(comp, comp_emb):
        data.append((torch.tensor(_emb), torch.tensor([0]), _comp))

    return data


def predict(struc_list, model_dir=MODEL_PATH, emb_path=EMB_PATH, is_cuda=True):

    is_cuda = torch.cuda.is_available()
    dataset_test = get_dataset(struc_list, emb_path)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False,
                             num_workers=0, collate_fn=collate_fn,
                             pin_memory=is_cuda)

    pred_list = []
    # Loop for all models
    for i in range(1, 101):

        modelpath = os.path.join(model_dir, 'checkpoint_bag_'+str(i)+'.pth.tar')

        if os.path.isfile(modelpath):
            # print("=> loading model params '{}'".format(modelpath))
            model_checkpoint = torch.load(modelpath,
                                          map_location=lambda storage, loc: storage)
            model_args = argparse.Namespace(**model_checkpoint['args'])
            # print("=> loaded model params '{}'".format(modelpath))
        else:
            print("=> no model params found at '{}'".format(modelpath))

        # build model
        model = Net(atom_fea_len=model_args.atom_fea_len,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h)

        if is_cuda:
            model.cuda()

        normalizer = Normalizer(torch.zeros(3))

        # optionally resume from a checkpoint
        if os.path.isfile(modelpath):
            # print("=> loading model '{}'".format(modelpath))
            checkpoint = torch.load(modelpath,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer.load_state_dict(checkpoint['normalizer'])
        else:
            print("=> no model found at '{}'".format(modelpath))

        preds = validate(test_loader, model, test=True, is_cuda=is_cuda)
        pred_list.append(preds)

    pred_array = np.array(pred_list)
    syn_score = pred_array.mean(axis=0)
    return syn_score


def validate(val_loader, model, test=True, is_cuda=True):
    if test:
        test_preds = []

    # switch to evaluate mode
    model.eval()

    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if is_cuda:
                input_var = Variable(input.cuda(non_blocking=True))
            else:
                input_var = Variable(input)

        # compute output
        output = model(input_var.float())

        # measure accuracy and record loss
        if test:
            test_pred = torch.exp(output.data.cpu())
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()

    return test_preds


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


if __name__ == '__main__':
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms_list = read('test.extxyz', index=":")
    adaptor = AseAtomsAdaptor()
    struc_list = [adaptor.get_structure(atoms) for atoms in atoms_list]
    sscore = predict(
        struc_list=struc_list,
        is_cuda=True,
    )
    print(sscore.mean())
