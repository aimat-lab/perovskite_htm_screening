from sklearn.linear_model import LinearRegression
from utils import generate_trainset, standardize_data, plot_scatter, leave_one_out_crossval


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


# forward optimization best r2, 8 features, r2: 0.456999140013172, bic: 381.1341164453419
single_task_best_features = ['has tertiary amine', 'rotation constant c', 'dipole', 'purity', 'aromatic bonds count', 'N count', 'log P', 'aromatic atoms count']
MASK = [(f in single_task_best_features) for f in features]
feat_order = [f for f in features if f in single_task_best_features]
X = X[..., MASK]

preds = []
ground = []
coefficients = {f: [] for f in feat_order}
coefficients['intercept'] = []
for xtrain, xtest, ytrain, ytest in leave_one_out_crossval(X, y, samples_composition, reject='both'):
    xtrain, ytrain, x_scaler, y_scaler = standardize_data(xtrain, ytrain)
    xtest = x_scaler.transform(xtest)
    reg = LinearRegression().fit(xtrain, ytrain)
    for f, c in zip(feat_order, reg.coef_[0]):
        coefficients[f].append(c)
    coefficients['intercept'].append(reg.intercept_)          
    preds.append(y_scaler.inverse_transform(reg.predict(xtest)).item())
    ground.append(ytest.item())
plot_scatter(preds, ground, save=True, name="linear_scatter.png")