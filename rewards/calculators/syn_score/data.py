from __future__ import print_function, division

import json
import os

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import re


def collate_pool(dataset_list):

    batch_struc_fea, batch_target = [], []
    batch_cif_ids = []
    for i, (struc_fea, target, cif_id)\
            in enumerate(dataset_list):
        batch_struc_fea.append(struc_fea)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
    return torch.stack(batch_struc_fea, dim=0), torch.stack(batch_target, dim=0), batch_cif_ids


def split_bagging(id_prop, bagging_size, folder):
    df = pd.read_csv(id_prop, header=None)

    # Split positive/unlabeled data
    exp = []
    vir = []
    for i in range(len(df)):
        if df[1][i] == 1:
            exp.append(df[0][i])
        elif df[1][i] == 0:
            vir.append(df[0][i])
        else:
            raise Exception("ERROR: prop value must be 1 or 0")

    positive = pd.DataFrame()
    positive[0] = exp
    positive[1] = [1 for _ in range(len(exp))]

    unlabeled = pd.DataFrame()
    unlabeled[0] = vir
    unlabeled[1] = [0 for _ in range(len(vir))]

    # Sample positive data for validation and training
    valid_positive = positive.sample(frac=0.2,random_state=1234)
    train_positive = positive.drop(valid_positive.index)

    os.makedirs(folder, exist_ok=True)

    print("Get splits for PU learning...")
    # Sample negative data for training
    for i in tqdm(range(bagging_size)):
        # Randomly labeling to negative
        negative = unlabeled.sample(n=len(positive[0]))
        valid_negative = negative.sample(frac=0.2,random_state=1234)
        train_negative = negative.drop(valid_negative.index)

        valid = pd.concat([valid_positive,valid_negative])
        valid.to_csv(os.path.join(folder, 'id_prop_bag_'+str(i+1)+'_valid.csv'), mode='w', index=False, header=False)

        train = pd.concat([train_positive,train_negative])
        train.to_csv(os.path.join(folder, 'id_prop_bag_'+str(i+1)+'_train.csv'), mode='w', index=False, header=False)

    # Generate unlabeled data
        test_unlabel = unlabeled.drop(negative.index)
        test_unlabel.to_csv(os.path.join(folder, 'id_prop_bag_'+str(i+1)+'_test-unlabeled.csv'), mode='w', index=False, header=False)


def bootstrap_aggregating(bagging_size, prediction=False):

    predval_dict = {}

    print("Do bootstrap aggregating for %d models.............." % (bagging_size))
    for i in range(1, bagging_size+1):
        if prediction:
            filename = 'test_results_prediction_'+str(i)+'.csv'
        else:
            filename = 'test_results_bag_'+str(i)+'.csv'
        df = pd.read_csv(os.path.join(filename), header=None)
        id_list = df.iloc[:,0].tolist()
        pred_list = df.iloc[:,2].tolist()
        for idx, mat_id in enumerate(id_list):
            if mat_id in predval_dict:
                predval_dict[mat_id].append(float(pred_list[idx]))
            else:
                predval_dict[mat_id] = [float(pred_list[idx])]

    print("Writing score file....")
    with open('test_results_ensemble_'+str(bagging_size)+'models.csv', "w") as g:
        g.write("comp,score,bagging")                                       # composition, score, # of bagging size

        for key, values in predval_dict.items():
            g.write('\n')
            g.write(key+','+str(np.mean(np.array(values)))+','+str(len(values)))
    print("Done")


def calculate_input(reduced_comp, dictionary):
    total = np.zeros(90)
    elem_num = 0
    for element in reduced_comp.split():
        split_str = re.sub('[^a-zA-Z]', '', element)
        split_int = re.sub('[^0-9]', '', element)
        elem_num += int(split_int)
        total = total + np.array(dictionary[split_str])*int(split_int)

    return total/elem_num


def preload_input(embedding_file, id_prop):
    input_dict = {}
    #/home/josh416/ML/cgcnn_node_feature_set/cgcnn_hd_rcut4_nn8.element_embedding.json
    with open(embedding_file) as f:
        dict = json.load(f)

    df = pd.read_csv(id_prop,header=None)

    print("Generates CGNF representation for the data.....")
    for i in tqdm(range(len(df))):
        composition = df[0][i]
        input_dict[composition] = calculate_input(composition, dict)

    return input_dict
