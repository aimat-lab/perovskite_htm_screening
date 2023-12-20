import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from RFM_model import RFM
from utils import generate_trainset, standardize_data


DATA_PATH = './data/dataset.csv'
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
X, y, samples_composition = generate_trainset(path=DATA_PATH, use_simulation=True, objective='PCE')

### ELIMINATE FEATURES THAT HAVE A SINGLE VALUES (constant) OR VERY NARROW DISTRIBUTIONS (spiked) ON THE TRAIN SET
####spiked = ['F', "O", 'S', "fr_C_O_noCOO", "has_Benzimidazole", "has_Benzothiophene", "has_Naphthalene", "has_Thiophene"]  
####constant = ['Cl', "has_C=NC", "has_CNC", "has_Pyrrole"] 
SPIKED = [5, 2, 4, 14, 25, 26, 27, 23]
CONSTANT = [6, 21, 22, 24]
NOT_GOOD = sorted(CONSTANT+SPIKED)
MASK_FEATURES = np.ones(X.shape[-1], dtype=bool)
MASK_FEATURES[NOT_GOOD] = False
X = X[..., MASK_FEATURES]
feat_order = [f for f, b in zip(features, MASK_FEATURES) if b]

preds = []
ground = []
importances = []
matrices = []
r2s = []
## run 1000 random splits
for _ in range(1000):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=None)
    xtrain, ytrain, x_scaler, y_scaler = standardize_data(xtrain, ytrain)
    xtest = x_scaler.transform(xtest)
    reg = RFM()
    reg.fit(
        xtrain.detach().numpy(), ytrain.detach().numpy(), reg=1e-3, num_iters=5,
        centering=True, verbose=False, diag_only=False,
    )
    matrix = reg.get_M()
    matrices.append(matrix/matrix.sum())
    M = np.diag(reg.get_M())
    importances.append(M / M.sum())        
    temp_preds = y_scaler.inverse_transform(reg.predict(xtest)).ravel()
    preds.append(temp_preds)
    ground.append(ytest.ravel())
    r2s.append(r2_score(ytest.ravel(), temp_preds))

preds = np.concatenate(preds, axis=0)
ground = np.concatenate(ground, axis=0)

matrix = 0
importance = 0
tot = 0
for i, M in enumerate(matrices):
    matrix += M
    importance += importances[i]
    tot += 1
matrix = matrix/tot
importance = importance/tot
feature_imp = pd.Series(importance, index=feat_order).sort_values(ascending=False)

plt.rcParams["figure.figsize"] = (10, 9)
ax = sns.barplot(x=feature_imp, y=feature_imp.index, color='royalblue', alpha=.7, edgecolor='black')
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)
plt.tight_layout()
plt.show()
plt.close()