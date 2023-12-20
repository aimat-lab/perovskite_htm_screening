import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from utils import generate_trainset, standardize_data, leave_one_out_crossval, plot_scatter
from GP_models import MTGPR


DATA_PATH = './data/dataset.csv'
ADD_OUTPUTS = ['Voc', 'Jsc', 'FF', 'D_V0', 'D_J0', 'contact_angle', 'PLQY_perov', 'PLQY_glass', 't1_perov', 't2_perov', 't1_glass', 't2_glass']
features = ["C count", "N count", "O count", "H count", "S count", 
            "F count", "Cl count", "atoms count", "atoms in ring", 
            "aromatic atoms count", "bonds count", "conjugated bonds count", 
            "aromatic bonds count", "rotatable bonds count", "carbonyl O (excl. COOH) count", 
            "exact molecular weight", "Morgan FP density", "fraction of SP3 C", "log P", 
            "molar refractivity", "has tertiary amine", "has secondary amine", "has imine", 
            "has thiophene", "has pyrrole", "has benzimidazole", "has benzothiophene", 
            "has naphthalene", "has biphenyl", "dipole", "homo level", "lumo level", 
            "homo/lumo gap", "total energy", "rotation constant a", "rotation constant b", 
            "rotation constant c", 'purity']

print('Generating trainset...')
X, y, samples_composition = generate_trainset(path=DATA_PATH, use_simulation=True, objective='PCE', add_labels=ADD_OUTPUTS)

### ELIMINATE FEATURES THAT HAVE A SINGLE VALUES (constant) OR VERY NARROW DISTRIBUTIONS (spiked) ON THE TRAIN SET
####spiked = ['F', "O", 'S', "fr_C_O_noCOO", "has_Benzimidazole", "has_Benzothiophene", "has_Naphthalene", "has_Thiophene"]  
####constant = ['Cl', "has_C=NC", "has_CNC", "has_Pyrrole"] 
SPIKED = [5, 2, 4, 14, 25, 26, 27, 23]
CONSTANT = [6, 21, 22, 24]
NOT_GOOD = sorted(CONSTANT+SPIKED)
MASK_FEATURES = np.ones(X.shape[-1], dtype=bool)
MASK_FEATURES[NOT_GOOD] = False
X = X[..., MASK_FEATURES]

task_labels = [
    'PCE', '$V_{oc}$', '$J_{sc}$', 'FF', '$D_{V_{0}}$', '$D_{J_{0}}$', 'CA', 
    '$PLQY_{perov}$', '$PLQY_{glass}$', '$t1_{perov}$', '$t2_{perov}$', 
    '$t1_{glass}$', '$t2_{glass}$',
]

xtrain, ytrain, _, _ = standardize_data(X, y)
regr = MTGPR(xtrain, ytrain)
regr.fit()

B = regr.model.covar_module.task_covar_module.covar_factor.detach().numpy()
v = regr.model.covar_module.task_covar_module.var.detach().numpy()
task_covar = np.matmul(B,B.T) + np.diag(v)

plt.figure(figsize = (22,18))
plt.rcParams['font.size'] = 30

print(np.abs(task_covar))
ax = sns.heatmap(
    np.abs(task_covar), 
    linewidth=0.5, 
    annot=True, 
    fmt=".2f",
    xticklabels=task_labels,
    yticklabels=task_labels,
    cbar_kws={'label':'abs'},
    )
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()
plt.close()