import numpy as np
import rdkit
import keras as ks
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.Draw
import pandas as pd
import sys
import json
import torch
import gpytorch
from torch.optim.lr_scheduler import StepLR
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, DotProduct, WhiteKernel, Matern, Exponentiation, ExpSineSquared, RationalQuadratic)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from reaction import ReactionAB
# from experiment import ExperimentalDataset
from features import MolFeatures
from theoretical import TheoSimulation

seed = 42
np.random.seed(seed)
# ks.utils.set_random_seed(seed)

sys.path.append("..")

use_simulation_data = True

label_names = ["PCE", "Jsc", "Voc", "FF"]
new_data = pd.read_csv("data/dataset.csv")

# For the model selection only the first initial data is availbable
new_data = new_data.iloc[:100, :]

samples, targets = [x.replace("(", "").replace(")", "").replace("'", "").replace(" ", "").split(",") for
                    x in new_data["AB"].values], new_data.loc[:, label_names].values

feat_cols = [
    "C",
    "N",
    "O",
    "H",
    "S",
    "F",
    # "Cl",
    "NumAtoms",
    "AtomIsInRing",
    "AtomIsAromatic",
    "NumBonds",
    "BondIsConjugated",
    "BondIsAromatic",
    "NumRotatableBonds",
    # "fr_C_O_noCOO",
    # "fr_Al_COO",
    # "fr_Ar_COO",
    # "fr_Al_OH",
    # "fr_Ar_OH",
    # "fr_NH2",
    # "fr_SH",
    # "fr_sulfide",
    # "fr_alkyl_halide"
    "ExactMolWt",
    "FpDensityMorgan3",
    "MolLogP",
    "MolMR",
    "FractionCSP3",
    "has_CN(C)C",
    # "has_cnc",
    # "has_C=NC",
    # "has_Thiophene",
    # "has_Pyrrole",
    # "has_Benzimidazole",
    # "has_Benzothiophene",
    # "has_Naphthalene",
    "has_Biphenyl"
]
theo_cols = [
    "dipole",
    "homo",
    "lumo",
    "gap",
    "energy",
    "a",
    "b",
    "c"
]

# Make reaction
reaction_engine = ReactionAB(file_name_a="Mol_Group_A.xlsx", file_name_b="Mol_Group_B.xlsx",
                             data_dir_path="./data")
mols = reaction_engine.run_combos(samples)
# reaction_engine.draw_to_pdf_products(samples)
# reaction_engine.save_to_mol_folder_for_simulation(samples, make_conformers=True, optimize_conformer=True, add_hydrogen=True)

# Make features
feature_generator = MolFeatures(descriptor_list=feat_cols)
features = feature_generator(mols)

if use_simulation_data:
    theo_generator = TheoSimulation(file_name="Theo_simu.xlsx", descriptor_list=theo_cols, data_dir_path="./data")
    _, theos = theo_generator.labels_for_combos(samples)
    features = np.concatenate([features, theos], axis=-1)

columns = {"ID": ["".join(x) for x in samples]}
columns.update({x: features[:, i] for i, x in enumerate(feat_cols)})
if use_simulation_data:
    columns.update({x: theos[:, i] for i, x in enumerate(theo_cols)})
columns.update({x: targets[:, i] for i, x in enumerate(label_names)})
columns.update({"MolFormular": [rdkit.Chem.rdMolDescriptors.CalcMolFormula(m) for m in mols]})

frame = pd.DataFrame(columns)
frame.to_excel("data/MolFeatures.xlsx", index=False)

# Ignore samples with smaller than 0.2 PCE.
ignore_samples_mask = targets[:, 2] > 0.2

selected_targets = targets[ignore_samples_mask]
selected_features = features[ignore_samples_mask]

# Std scaling needs to be replaced with better scaler.
y_scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
x_scaler = StandardScaler(with_std=True, with_mean=True, copy=True)

scaled_targets = y_scaler.fit_transform(selected_targets)
scaled_features = x_scaler.fit_transform(selected_features)

# For validation, use a KFold() split.
kf = KFold(n_splits=10, random_state=None, shuffle=True)
split_indices = kf.split(X=scaled_features)

print("Fitting model to data...")

fit_stats = []
validation_stats = []
models_fitted = {
    "Gauss": [],
    "Kernel": [],
    "NN": [],
    "Linear": [],
    "RF": []
}

for train_index, test_index in split_indices:
    ytrain = scaled_targets[train_index]
    ytest = scaled_targets[test_index]
    xtrain = scaled_features[train_index]
    xtest = scaled_features[test_index]

    for model_type in models_fitted.keys():
        print(model_type)

        if model_type == "Linear":
            model = LinearRegression().fit(xtrain, ytrain)

        if model_type == "Gauss":
            kernel = Matern(length_scale=1.0) + WhiteKernel() + DotProduct()
            model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(xtrain, ytrain)

        elif model_type == "MGP":

            class MultitaskGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super().__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.MultitaskMean(
                        gpytorch.means.ConstantMean(),
                        num_tasks=train_y.shape[-1]
                    )
                    self.covar_module = gpytorch.kernels.MultitaskKernel(
                        gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]),
                        num_tasks=train_y.shape[-1],
                        rank=1
                    )

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


            class MTGPR:
                def __init__(self, train_x, train_y):
                    if isinstance(train_x, np.ndarray):
                        self.train_x = torch.tensor(train_x).float()
                    else:
                        self.train_x = train_x
                    if isinstance(train_y, np.ndarray):
                        self.train_y = torch.tensor(train_y).float()
                    else:
                        self.train_y = train_y
                    self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[-1])
                    self.model = MultitaskGPModel(self.train_x, self.train_y, self.likelihood)
                    self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes likelihood parameters
                    self.scheduler = StepLR(self.optimizer, step_size=40, gamma=0.7)

                def fit(self, training_iterations=250):
                    self.model.train()
                    self.likelihood.train()
                    for i in range(training_iterations):
                        self.optimizer.zero_grad()
                        output = self.model(self.train_x)
                        loss = -self.mll(output, self.train_y)
                        loss.backward()
                        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                        self.optimizer.step()
                        if self.optimizer.param_groups[0]['lr'] > 1e-2:
                            self.scheduler.step()
                        if self.optimizer.param_groups[0]['lr'] < 1e-2:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = 1e-1

                def predict(self, test_x, return_std=False):
                    if isinstance(test_x, np.ndarray):
                        torch_test_x = torch.tensor(test_x).float()
                    else:
                        torch_test_x = test_x
                    self.model.eval()
                    self.likelihood.eval()
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        predictions = self.likelihood(self.model(torch_test_x))
                        mean = predictions.mean
                        if return_std:
                            lower, upper = predictions.confidence_region()
                            return mean, (upper - lower) / 4
                        else:
                            return mean

            print("Fitting model...")
            model = MTGPR(xtrain, ytrain)
            model.fit()

        elif model_type == "Kernel":
            params = {'alpha': 0.1, 'gamma': None, 'kernel': 'laplacian'}
            model = KernelRidge(**params).fit(xtrain, ytrain)

        elif model_type == "RF":
            params = {}
            model = RandomForestRegressor(**params).fit(xtrain, ytrain)

        else:
            model = ks.Sequential()
            model.add(ks.layers.Input(shape=[features.shape[-1]]))
            model.add(ks.layers.BatchNormalization())
            model.add(ks.layers.Dense(100,
                                            activation=ks.layers.LeakyReLU(alpha=0.05),
                                            kernel_regularizer=ks.regularizers.L1(1e-8)))
            model.add(ks.layers.Dense(100,
                                            activation=ks.layers.LeakyReLU(alpha=0.05),
                                            kernel_regularizer=ks.regularizers.L1(1e-8)))
            model.add(ks.layers.Dense(ytrain.shape[-1]))

            model.compile(optimizer=ks.optimizers.Adam(learning_rate=1e-4),
                          loss='mae', metrics=['mae', 'mse', 'mape'])

            history = model.fit(
                x=xtrain, y=ytrain,
                batch_size=16, epochs=100, verbose=1, callbacks=None, shuffle=True,
                validation_data=(xtest, ytest))
            fit_stats.append(history)

        predicted = model.predict(xtest)
        predicted = y_scaler.inverse_transform(predicted)
        test_labels_rescaled = y_scaler.inverse_transform(ytest)
        models_fitted[model_type].append([predicted, test_labels_rescaled])


# Plot the GP results
validation_stats = models_fitted["Gauss"]
r2_stats = [
    r2_score(np.concatenate([x[1][:, i] for x in validation_stats], axis=0),
             np.concatenate([x[0][:, i] for x in validation_stats], axis=0)) for
    i in range(scaled_targets.shape[-1])]
# r2_stats = [r2_score(x[1],x[0]) for x in validation_stats]  # Not stable if r2 < 0
mae_stats = [np.mean(np.abs(x[1] - x[0]), axis=0) for x in validation_stats]

fig, axsg = plt.subplots(1, 4, figsize=(17.25, 3.75))
titles = ["PCE [%]",  r"J$_{sc}$ [mA/cm$^2$]", "Voc [V]", "FF [%]"]
axs = axsg.flatten()
for j in range(scaled_targets.shape[-1]):
    axs[j].plot([np.amin(selected_targets[:, j]), np.amax(selected_targets[:, j])],
                [np.amin(selected_targets[:, j]), np.amax(selected_targets[:, j])], "--",
                label=r"r$^2$: {0:0.3f}, MAE: {1:0.4f} $\pm$ {2:0.3f}".format(r2_stats[j],
                                                                              np.mean(mae_stats, axis=0)[j],
                                                                              np.std(mae_stats, axis=0)[j]),
                c="indianred")
    for i in range(len(validation_stats)):
        x_pred, y_actual = validation_stats[i]
        axs[j].scatter(x_pred[:, j], y_actual[:, j], alpha=0.65, c="royalblue")

    axs[j].grid(True)
    axs[j].set_title(titles[j])
    axs[j].legend(loc="lower left")
    axs[j].set_xlabel("Predicted")
    if j == 0:
        axs[j].set_ylabel("True")

axs[0].text(
    -0.2, 1.0, 'C', fontsize=18, weight="bold",
    transform=axs[0].transAxes,
)
plt.savefig("GP_labels.png", bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot training curve
plt.figure()
for x in fit_stats:
    plt.plot(np.array(x.history["mae"]) * np.mean(y_scaler.scale_), c="blue")
for x in fit_stats:
    plt.plot(np.array(x.history["val_mae"]) * np.mean(y_scaler.scale_), c="red")
plt.xlabel('ML Prediction')
plt.ylabel('Measured device property')
plt.savefig("learning_curve_nn.png", bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot models
fig, axsg = plt.subplots(1, 5, figsize=(18.25, 3.75), sharey=True)
fig.subplots_adjust(wspace=0.)
axs = axsg.flatten()
full_title = {"Gauss": "Gaussian Process", "Kernel": "Kernel Ridge", "NN": "Neural Network", "Linear": "Linear",
              "RF": "Random Forest"}
for j, model_type in enumerate(models_fitted):

    validation_stats = models_fitted[model_type]
    r2_stats = [
        r2_score(np.concatenate([x[1][:, i] for x in validation_stats], axis=0),
                 np.concatenate([x[0][:, i] for x in validation_stats], axis=0)) for
        i in range(scaled_targets.shape[-1])]
    # r2_stats = [r2_score(x[1],x[0]) for x in validation_stats]  # Not stable if r2 < 0
    mae_stats = [np.mean(np.abs(x[1] - x[0]), axis=0) for x in validation_stats]

    axs[j].plot([np.amin(selected_targets[:, 0]), np.amax(selected_targets[:, 0])],
                [np.amin(selected_targets[:, 0]), np.amax(selected_targets[:, 0])], "--",
                label=r"r$^2$: {0:0.3f}, MAE: {1:0.4f} $\pm$ {2:0.3f}".format(r2_stats[0],
                                                                              np.mean(mae_stats, axis=0)[0],
                                                                              np.std(mae_stats, axis=0)[0]),
                c="indianred")
    for i in range(len(validation_stats)):
        x_pred, y_actual = validation_stats[i]
        axs[j].scatter(x_pred[:, 0], y_actual[:, 0], alpha=0.65, c="royalblue")

    axs[j].grid(True)
    axs[j].set_title(full_title[model_type])
    axs[j].legend(loc="lower left")
    axs[j].set_xlabel("Predicted PCE [%]")
    if j==0:
        axs[j].set_ylabel("True PCE [%]")

axs[0].text(
    -0.2, 1.0, 'B', fontsize=18, weight="bold",
    transform=axs[0].transAxes,
)

plt.savefig("all_models.png", bbox_inches='tight', pad_inches=0.1)
plt.show()

