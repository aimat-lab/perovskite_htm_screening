import numpy as np
import torch
import gpytorch

from torch.optim.lr_scheduler import StepLR
from botorch import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood



def at_least_2dim(x):
    if len(x.shape)<2:
        x = x.reshape(-1, 1)
    return x


class GPR:
    def __init__(self, train_x, train_y, noise_free=False):
        train_x = at_least_2dim(train_x)
        train_y = at_least_2dim(train_y)
        if isinstance(train_x, np.ndarray):
            train_x, train_y = torch.tensor(train_x).float(), torch.tensor(train_y).float()
        self.initialize_model(train_x, train_y, noise_free)
        
    def initialize_model(self, train_x, train_y, noise_free):
        models = []
        for i in range(train_y.shape[-1]):
            ## mean_module : ConstantMean, likelihood : GaussianLikelihood with inferred noise level
            if noise_free:
                train_Yvar = torch.full_like(train_y[..., i : i + 1], 1e-3)
                models.append(FixedNoiseGP(train_x, train_y[..., i : i + 1], train_Yvar=train_Yvar, 
                                            covar_module=gpytorch.kernels.ScaleKernel(
                                                                            gpytorch.kernels.RBFKernel(
                                                                                ard_num_dims=train_x.shape[-1]))
                                                                                                        ))
            else:
                models.append(SingleTaskGP(train_x, train_y[..., i : i + 1], 
                                            covar_module=gpytorch.kernels.ScaleKernel(
                                                                            gpytorch.kernels.RBFKernel(
                                                                                ard_num_dims=train_x.shape[-1]))
                                                                                                        ))
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
    
    def fit(self):
        fit_gpytorch_model(self.mll)
        
    def predict(self, x, return_posterior=False, no_grad=True):
        if isinstance(x, np.ndarray):
            x = torch.tensor(at_least_2dim(x)).float()
        if no_grad:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = self.model.posterior(x)
        else:
            posterior = self.model.posterior(x)
        if return_posterior:
            return posterior
        else:
            mean, var = posterior.mean, posterior.variance
            std = torch.sqrt(var)
            return mean, std
      

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
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[-1])
        self.model = MultitaskGPModel(train_x, train_y, self.likelihood)
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
            #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            self.optimizer.step()
            if self.optimizer.param_groups[0]['lr'] > 1e-2:
                self.scheduler.step()
            if self.optimizer.param_groups[0]['lr'] < 1e-2:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-1
    
    def predict(self, test_x):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
            return mean, (upper-lower)/4






