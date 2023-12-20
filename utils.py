import sys
import json
import rdkit
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from reaction import ReactionAB
from features import MolFeatures
from theoretical import TheoSimulation



def at_least_2d(x):
    if len(x.shape)<2:
        x = x[np.newaxis, ...]
    return x


######################################### UPLOAD EXPERIMENTAL AND SIMULATION DATA ######################################################################

def produce_mol_dataset(name: str = 'dataset', n_mol : int = 2000, rng = None):
    PROBLEMS = {'A263', 'A435', 'A439', 'A440', 'A485', 'A486', 'A518', 'A530', 'A546', 'A630', 'A688', 'A689', 'A690', 'A879', 'A1115'}
    LIMIT = 100000
    
    if rng==None:
        rng = np.random.default_rng()

    df = pd.read_excel('./data/Mol_Group_A.xlsx', sheet_name=0, header=0)
    As = df.loc[:, 'ID']
    df = pd.read_excel('./data/Mol_Group_B.xlsx', sheet_name=0, header=0)
    Bs = df.loc[:, 'ID']

    As = list(set(As)-PROBLEMS)
    n_as = min(len(As), n_mol)
    n_bs = min(len(Bs), n_mol)
    As = rng.choice(As, n_as, replace=False)
    Bs = rng.choice(Bs, n_bs, replace=False)

    samples = [(a, b) for a, b in product(As, Bs) if a not in PROBLEMS]

    if len(samples)>LIMIT:
        n_chunks = int(len(samples)/LIMIT) + 1
    else:
        n_chunks = 1
    
    mols = []
    data = {}
    for i in range(n_chunks):   
        chunk = samples[i*LIMIT:(i+1)*LIMIT]
        mols = ReactionAB().run_combos(chunk)

        keys = [a+b for a,b in chunk]	
        data = dict(zip(keys, mols))

        np.save('./data/Mols/'+name+'_chunk'+str(i)+'.npy', data)
        
        mols = []
        data = {}

def get_features(mols, samples, use_simulation_data=False):
    feature_generator = MolFeatures()
    features = feature_generator(mols)
    if use_simulation_data:
        _, theos = TheoSimulation().labels_for_combos(samples)
        features = np.concatenate([features, theos], axis=-1)
    return features
    

def generate_trainset(path='./data/dataset.csv', use_simulation=False, objective='PCE', add_labels=None):
    df = pd.read_csv(path)
    labels = [objective]
    if not add_labels is None:
        labels += add_labels
    targets = df[labels].to_numpy()
    if len(targets.shape)<2:
        target = target[..., np.newaxis]
    samples = []
    for ab in df['AB']: 
        a, b = ab[1:-1].split(',')
        samples.append((a[1:-1], b[2:-1]))
    
    reaction_engine = ReactionAB(file_name_a="Mol_Group_A.xlsx", file_name_b="Mol_Group_B.xlsx", data_dir_path='data')
    mols = reaction_engine.run_combos(samples)
    features = get_features(mols, samples=samples, use_simulation_data=use_simulation)
    
    ### add purity
    purity = pd.read_excel('./data/ID-1-purity.xlsx')['purity (%)'].to_numpy()[:len(features)]
    features = np.concatenate((features, purity[..., np.newaxis]), axis=-1)
    

    # Ignore samples with smaller than 0.2 PCE.
    ignore_samples_mask = targets[:, 0] > 0.2
    selected_targets = targets[ignore_samples_mask]
    selected_features = features[ignore_samples_mask]
    return selected_features, selected_targets, np.asarray(samples)[ignore_samples_mask]



#######################################################  PREPROCESSING #####################################################################################  
    
def standardize_data(X, y):
    x_scaler = StandardScaler(with_std=True, with_mean=True, copy=True).fit(X) 
    y_scaler = StandardScaler(with_std=True, with_mean=True, copy=True).fit(y)
    scaled_x = torch.tensor(x_scaler.transform(X)).float()
    scaled_y = torch.tensor(y_scaler.transform(y)).float()
    return scaled_x, scaled_y, x_scaler, y_scaler


#######################################################  CROSS-VALIDATION  ###################################################################################


def leave_one_out_crossval(X, y, samples_composition, reject='both'):
    samples_composition = np.array([[a, b] for (a, b) in samples_composition])
    for idx, (A, B) in enumerate(samples_composition):
        if reject=='both':
            mask_seen_fragments = np.logical_and(samples_composition[:, 0]!=A, samples_composition[:, 1]!=B)
        elif reject=='A':
            mask_seen_fragments = samples_composition[:, 0]!=A
        elif reject=='B':
            mask_seen_fragments = samples_composition[:, 1]!=B
        xtrain = at_least_2d(X[mask_seen_fragments])
        ytrain = at_least_2d(y[mask_seen_fragments])
        xtest = at_least_2d(X[idx])
        ytest = at_least_2d(y[idx])
        yield xtrain, xtest, ytrain, ytest


############################################ PLOTS ###########################################

def plot_scatter(preds, targets, name='scatter_plot', save=False):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(
        [np.amin(targets),np.amax(targets)],
        [np.amin(targets),np.amax(targets)], 
        label=r"R$^2$: {0:0.3f}".format(r2_score(targets, preds)), 
        c="red", 
        linestyle='--',
    )
    ax.scatter(preds, targets, alpha=0.65, c="royalblue", s=80)   
    ax.legend(loc="upper left", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=18)
    ax.set_ylabel("True", fontsize=18)
    plt.title(f"PCE [%]", fontsize=18)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(f"{name}.png", bbox_inches = 'tight', pad_inches=0.1)
    else:
        plt.show()  
    plt.close() 

