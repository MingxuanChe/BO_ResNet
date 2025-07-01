import pathlib
import sys
from copy import deepcopy

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm
from gpytorch.kernels import RBFKernel, ScaleKernel
from scipy.optimize import minimize
from scipy.stats import qmc

import wandb

# check whether helper.py is in the system path
try:
    from helper import Options, bo_config, device, set_seed
except ImportError:
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from helper import Options, bo_config, device, set_seed

# set seaborn style
sns.set_theme(style='darkgrid', palette='deep')


# NOTE: we let BO consider maximization problem


def get_sobol_init(num_sample, num_dim=1, bounds=None):
    """Generate Sobol sequence samples in the specified bounds"""
    # NOTE: sobol sequence need to set the seed explicitly
    # Add explicit seed for reproducibility
    sobol = qmc.Sobol(d=num_dim, seed=bo_config.seed)
    samples = sobol.random(n=num_sample)
    # will ample in [0, 1) by default
    if bounds is not None:
        low, high = bounds[:, 0], bounds[:, 1]
        samples = low + (high - low) * samples
        assert np.all(samples >= low) and np.all(samples < high), \
            'Samples are not within the specified range.'
    return samples


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y,
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                 mean_module=gpytorch.means.ConstantMean(),
                 kernel='RBF',
                 config=None,
                 ):
        """Initialize the ExactGP model"""
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        if kernel == 'RBF':
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel == 'Matern':
            self.covar_module = ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5))

        # add priors
        self._add_priors(config)

    def _add_priors(self, config):
        """Add priors to kernel parameters based on config"""

        print('Before setting priors, kernel parameters are:')
        print(
            f'Lengthscale: {self.covar_module.base_kernel.lengthscale.item()}')
        print(f'Outputscale: {self.covar_module.outputscale.item()}')
        print(f'Noise: {self.likelihood.noise.item()}')

        if hasattr(config, 'lengthscale_init_mean'):
            self.covar_module.base_kernel.initialize(
                lengthscale=torch.tensor(config.lengthscale_init_mean)
            )

        if hasattr(config, 'outputscale_init_mean'):
            self.covar_module.initialize(
                outputscale=torch.tensor(config.outputscale_init_mean)
            )
        if hasattr(config, 'noise_init_var'):
            self.likelihood.initialize(
                noise=torch.tensor(config.noise_init_var)
            )
            print('kernel parameters initialized with config values:')
            print(
                f'Lengthscale: {self.covar_module.base_kernel.lengthscale.item()}')
            print(f'Outputscale: {self.covar_module.outputscale.item()}')
            print(f'Noise: {self.likelihood.noise.item()}')

        # lengthscale prior
        if hasattr(config, 'lengthscale_prior_mean') and \
                hasattr(config, 'lengthscale_prior_std'):
            self.covar_module.base_kernel.register_prior(
                'lengthscale_prior',
                gpytorch.priors.NormalPrior(
                    config.lengthscale_prior_mean,
                    config.lengthscale_prior_std
                ),
                'lengthscale',
            )

        # outputscale prior
        if hasattr(config, 'outputscale_prior_mean') and \
                hasattr(config, 'outputscale_prior_std'):
            self.covar_module.register_prior(
                'outputscale_prior',
                gpytorch.priors.NormalPrior(
                    config.outputscale_prior_mean,
                    config.outputscale_prior_std
                ),
                'outputscale',
            )

        # Noise prior
        if hasattr(config, 'noise_prior_mean') and \
           hasattr(config, 'noise_prior_std'):
            self.likelihood.noise_covar.register_prior(
                'noise_prior',
                gpytorch.priors.NormalPrior(
                    config.noise_prior_mean,
                    config.noise_prior_std,
                ),
                'noise',
            )

    def forward(self, x):
        """Forward pass"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianOptimization:
    def __init__(self,
                 unknown_function,
                 bo_config,
                 ):
        """
        Initialize the Bayesian Optimization model

        Args:
            unknown_function (callable): The function to be optimized.
            bo_config (Options): Configuration with hyperparameters for BO.
        """
        self.unknown_function = unknown_function
        self.config = bo_config
        self.kernel = self.config.kernel
        self.search_space = self.config.hp_search_space
        self.sst = self.config.hp_search_space_transformation
        if self.sst is not None:
            self.original_search_space = deepcopy(self.search_space)
            self.search_space = self.transform_search_space(self.search_space)

        self.budget = self.config.budget
        self.remaining_budget = deepcopy(self.budget)

        # sample history
        self.sampled_x = []
        self.sampled_y = []
        # store the model state dicts for easy access later
        self.model_history = []

    def transform_search_space(self, x, inverse=False):
        """
        Transform the search
        """
        if self.sst == 'log10':
            transformed_x = np.log10(x) if not inverse else 10 ** x
        else:
            raise ValueError(
                f'Search space transformation {self.sst} not implemented.')
        return transformed_x

    def get_incumbent(self):
        """
        Get the current incumbent value and its corresponding x
        """
        if len(self.sampled_y) == 0:
            return None, None

        incumbent_index = torch.argmax(torch.tensor(self.sampled_y)).item()
        incumbent_x = self.train_x[incumbent_index].item()
        incumbent_y = self.train_y[incumbent_index].item()

        if self.sst is not None:
            incumbent_x = self.transform_search_space(
                incumbent_x, inverse=True)

        return incumbent_index, incumbent_x, incumbent_y

    def initialize_bayesian_optimization(self,
                                         method='sobol',
                                         init_budget=3):

        if method == 'sobol' and init_budget > 0:
            sobol_samples = \
                get_sobol_init(num_sample=init_budget,
                               bounds=self.search_space).reshape(-1).tolist()
            # for sample in sobol_samples:
            pbar = tqdm.tqdm(
                sobol_samples,
                desc='Initializing BO with Sobol samples',
                unit='samples'
            )
            for i, sample in enumerate(pbar):
                # for i, sample in enumerate(sobol_samples):
                sample_original = \
                    self.transform_search_space(sample, inverse=True) \
                    if self.sst is not None else sample
                print(
                    f'Evaluating sample {i + 1}/{init_budget}: {sample_original}')
                y_value, model_state_dict = self.unknown_function(
                    sample_original)
                self.sampled_x.append(sample)
                self.sampled_y.append(y_value)
                self.model_history.append(model_state_dict)

        else:
            raise ValueError(f'Unknown initialization method: {method}')

        self.train_x = torch.tensor(
            self.sampled_x, dtype=torch.float32).reshape(-1)
        self.train_y = torch.tensor(
            self.sampled_y, dtype=torch.float32).reshape(-1)

        # find the current incumbent
        _, incumbent_x, incumbent_y = self.get_incumbent()
        print(f'Initial incumbent x: {incumbent_x}, y: {incumbent_y}\n')
        # subtract the budget
        self.remaining_budget = self.budget - init_budget

    def fit(self):
        """
        Fit the model to the training data
        """
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp_model.parameters(),
                                     lr=self.config.gp_learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.gp_model)

        for _ in range(self.config.gp_num_epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.train_x)
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            optimizer.step()
            if _ % (self.config.gp_num_epochs / 5) == 0:
                print(f'Epoch {_}: GP Loss = {loss.item()}')

        print('Kernel parameters after fitting:')
        print(
            f'Lengthscale: {self.gp_model.covar_module.base_kernel.lengthscale.item()}')
        print(f'Outputscale: {self.gp_model.covar_module.outputscale.item()}')
        print(f'Noise: {self.likelihood.noise.item()}')

    def expected_improvment(self, mean, std, incumbent, xi=0.0):
        '''
        Calculate the expected improvement (EI) given the mean, std, and incumbent value.
        NOTE: In the online lecture, minimization is used;
              in the BO tutorial paper + Murphy's book, maximization is used.
        NOTE: In this task, we can naturally consider maximization
        '''
        # avoid division by zero
        if torch.isclose(std, torch.tensor(0.0)):
            return 0.0

        z = (mean - incumbent - xi) / std
        exploitation_term = (mean - incumbent - xi) * \
            torch.distributions.Normal(0, 1).cdf(z)
        exploration_term = std * \
            torch.distributions.Normal(0, 1).log_prob(z).exp()

        acquisition_value = exploitation_term + exploration_term
        return acquisition_value

    def acquisition_value_at(self, x):
        """
        Calculate the acquisition value at a given point x
        """
        # make sure x is a numpy array
        if isinstance(x, np.ndarray):
            x = x.item()
        elif isinstance(x, torch.Tensor):
            x = x.item()

        x_tensor = torch.tensor([[x]], dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp_model.eval()
            self.likelihood.eval()
            pred = self.likelihood(self.gp_model(x_tensor))
            mean = pred.mean
            std = pred.stddev
            incumbent = torch.max(self.train_y)
            acquisition_value = self.expected_improvment(mean, std, incumbent)

            return acquisition_value

    def optimize_acquisition(self, acuisition_func):
        '''
        maximize the acquisition function with
        scipy.optimize.minimize (learned from BoTorch docs
        cf. https://botorch.org/docs/optimization#using-scipy-optimizers-on-tensors)
        '''
        best_x = None
        best_acquisition_value = -np.inf

        for _ in range(self.config.acquisitiion_optimization_budget):
            x_0 = np.random.uniform(self.search_space[:, 0],
                                    self.search_space[:, 1])
            res = minimize(lambda x: -acuisition_func(x),
                           x0=x_0,
                           bounds=self.search_space.tolist(),
                           method='L-BFGS-B')
            # NOTE: result value neesd to be negated
            if res.success and -res.fun > best_acquisition_value:
                best_x = res.x[0]
                best_acquisition_value = -res.fun

        if best_x is None:
            raise ValueError('No valid acquisition point found.')

        return best_x, best_acquisition_value

    def bayesian_optimization(self, init_budget=3):
        """
        Perform Bayesian Optimization

        Args:
            init_budget (int): Initial budget for the Sobol sequence initialization.
        Returns:
            tuple: The best found point and its corresponding value.

        """
        # 1. initialize the dataset
        self.initialize_bayesian_optimization(
            method=self.config.bo_init_method,
            init_budget=init_budget)
        # 2. loop until budget
        pbar = tqdm.tqdm(range(init_budget, self.budget),
                         desc='BO Progress',
                         unit='iterations',)
        # for i in range(init_budget, self.budget)
        for i in pbar:
            print(f'Starting iteration {i + 1}')
            # GP
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_model = ExactGP(self.train_x, self.train_y,
                                    self.likelihood, kernel=self.kernel,
                                    config=self.config)

            # 3. fit the GP model
            self.fit()
            # 4. optimize the acquisition function
            next_x, _ = self.optimize_acquisition(self.acquisition_value_at)

            self.plot_iteration_results(i, next_x)
            # 5. query the unknown function
            next_x_original = self.transform_search_space(next_x, inverse=True) \
                if self.sst is not None else next_x
            print(f'Evaluating next x: {next_x_original}')
            next_y, model_state_dict = self.unknown_function(next_x_original)
            # 6. update the training data
            self.sampled_x.append(next_x)
            self.sampled_y.append(next_y)
            self.model_history.append(model_state_dict)

            self.train_x = torch.tensor(self.sampled_x, dtype=torch.float32)
            self.train_y = torch.tensor(self.sampled_y, dtype=torch.float32)

            print(f'Iteration {i + 1}: Next x = {next_x}, Next y = {next_y}')
            # update the incumbent
            _, incumbent_x, incumbent_y = self.get_incumbent()
            print(f'Current incumbent x: {incumbent_x}, y: {incumbent_y}\n')

            # Log to wandb
            try:
                wandb.log({
                    'iteration': i + 1,
                    'next_x': next_x_original,
                    'next_y': next_y,
                    'incumbent_x': incumbent_x,
                    'incumbent_y': incumbent_y,
                    'num_samples': len(self.sampled_x)
                })
            except Exception:
                pass

        # return the best found point
        _, best_x, best_y = self.get_incumbent()
        print(f'Best x: {best_x}, Best y: {best_y}')

        # final log
        try:
            wandb.log({
                'bo_final_best_x': best_x,
                'bo_final_best_y': best_y,
                'bo_total_evaluations': len(self.sampled_x),
                'bo_improvement': best_y - min(self.sampled_y),
                'bo_sampled_x': self.sampled_x,
                'bo_sampled_y': self.sampled_y
            })
        except Exception:
            pass

        return best_x, best_y

    def plot_iteration_results(self, i, next_x, s=3):
        """
        Plot the results of the Bayesian Optimization iterations
        """
        x = np.linspace(self.search_space[0, 0], self.search_space[0, 1], 100)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp_model.eval()
            self.likelihood.eval()
            x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            pred = self.likelihood(self.gp_model(x_tensor))
            mean = pred.mean.numpy()
            std = pred.stddev.numpy()

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        # plot the objective function (ommited for simplicity)
        # y = np.array([self.unknown_function(xi)[0]
        #               for xi in x])
        # ax[0].plot(x, y, label='Objective Function', color='r')
        ax[0].plot(x, mean, label='GP Mean', color='blue')
        ax[0].fill_between(x, mean - s * std, mean + s * std,
                           color='lightblue', alpha=0.5, label=f'{s}-$\sigma$ Confidence Interval')
        ax[0].scatter(self.sampled_x, self.sampled_y,
                      color='k', label='Sampled Points')
        ax[0].axvline(x=next_x, color='orange', linestyle='--',
                      label='Next Sample Point')
        ax[0].set_title('Objective Function with Sampled Points')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('f(x)')
        ax[0].legend()
        ax[0].grid()

        # plot the acquisition function
        acquisition_values = [self.acquisition_value_at(
            torch.tensor([[xi]], dtype=torch.float32)).item() for xi in x]
        ax[1].plot(x, acquisition_values,
                   label='Acquisition Function', color='green')
        ax[1].scatter(next_x, self.acquisition_value_at(
            torch.tensor([[next_x]], dtype=torch.float32)).item(),
            color='orange', label='Next Sample Point', zorder=5
        )
        ax[1].axvline(x=next_x, color='orange', linestyle='--',
                      label='Next Sample Point')
        ax[1].set_title('Acquisition Function with Sampled Points')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('Acquisition Value')
        ax[1].legend()
        ax[1].grid()

        if self.sst is not None:
            # Create secondary x-axis at the top to show the
            # original search space values
            for axis in ax:
                ax_top = axis.secondary_xaxis('top')
                ax_ticks_values = axis.get_xticks()
                original_ticks_values = self.transform_search_space(
                    ax_ticks_values, inverse=True)
                ax_top.set_xticks(ax_ticks_values)
                ax_top.set_xticklabels(
                    [f'{t:.5f}' for t in original_ticks_values])
                ax_top.set_xlabel('x (original scale)')
                axis.set_xlabel('x (log scale)')

        fig.suptitle(f'Bayesian Optimization Iteration {i + 1}', fontsize=16)
        fig.tight_layout()

        # save the figure
        fig_path = self.config.results_dir / \
            f'bo_iterations_results_{i + 1}.png'
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)

    def plot_bo_history(self, s=3):
        """
        Plot the history of the Bayesian Optimization
        """
        if len(self.sampled_x) == 0 or len(self.sampled_y) == 0:
            print('No sampled points to plot.')
            return

        best_idx, best_x, best_y = self.get_incumbent()
        x = np.linspace(self.search_space[0, 0], self.search_space[0, 1], 100)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp_model.eval()
            self.likelihood.eval()
            x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            pred = self.likelihood(self.gp_model(x_tensor))
            # get the mean and std of the GP model
            mean = pred.mean.numpy()
            std = pred.stddev.numpy()

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(x, mean, label='GP Mean', color='blue')
        ax[0].fill_between(x, mean - s * std, mean + s * std,
                           color='lightblue', alpha=0.5, label='3-$\sigma$ Confidence Interval')
        ax[0].scatter(self.sampled_x, self.sampled_y,
                      color='k', label='Sampled Points')
        best_x_original = best_x
        if self.sst is not None:
            # transform the best_x back to the search space for plotting reasons
            best_x_original = self.transform_search_space(best_x)
        ax[0].scatter(best_x_original, best_y, marker='*', s=200,
                      color='orange', label='Current Best Point', zorder=5)
        ax[0].axvline(x=best_x_original, color='orange', linestyle='--',
                      label='Current Best Point Line')

        ax[0].set_title('Surrogatem Model with Sampled Points')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('f(x)')
        ax[0].legend()
        ax[0].grid()

        if self.sst is not None:
            # Create secondary x-axis at the top to show the
            # original search space values
            ax_top = ax[0].secondary_xaxis('top')
            ax_ticks_values = ax[0].get_xticks()
            original_ticks_values = self.transform_search_space(
                ax_ticks_values, inverse=True)
            ax_top.set_xticks(ax_ticks_values)
            ax_top.set_xticklabels([f'{t:.5f}' for t in original_ticks_values])
            ax_top.set_xlabel('x (original scale)')
            ax[0].set_xlabel('x (log scale)')

        # plot sample history along iterations
        ax[1].plot(range(1, len(self.sampled_x) + 1),
                   self.sampled_y, marker='o', label='Sampled Points', color='k')
        # plot shaded area before the initial budget
        ax[1].fill_between(range(1, self.config.init_budget + 1),
                           [min(self.sampled_y)] * (self.config.init_budget),
                           [max(self.sampled_y)] * (self.config.init_budget),
                           color='gray', alpha=0.5, label='Initial Budget Area')
        ax[1].axhline(y=best_y, color='orange', linestyle='--',
                      label='Current Best Value')
        ax[1].scatter(best_idx + 1, best_y, marker='*', s=200,
                      color='orange', label='Current Best Point', zorder=5)
        ax[1].set_title('Optimization History')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('f(x)')
        # ensure the x-ticks are integers
        ax[1].set_xticks(range(1, len(self.sampled_x) + 1))
        ax[1].legend()
        ax[1].grid()

        fig.suptitle('Bayesian Optimization History', fontsize=16)
        fig.tight_layout()

        # save the figure
        fig_path = self.config.results_dir / 'bo_history.png'
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)


def simple_objectve_function(x):
    """
    An example objective function to test BO

    Args:
        x (float or np.ndarray): Input value(s) for the function.
    Returns:
        tuple: A tuple containing the function value and None (for compatibility).
        model_state_dict: placeholder for model state dict
    """
    model_state_dict = None
    return -(x - 0.5)**2 + 3 * np.sin(x * 3) + 0.5, model_state_dict


def main_bo(config: Options):
    '''
    Main function to run the Bayesian Optimization with a simple objective function.
    Args:
        config (Options): Configuration object containing hyperparameters for BO.
        For details, see the docstring in helper.py -> Options.
    '''
    set_seed(config.seed, device=device)
    bo = BayesianOptimization(unknown_function=simple_objectve_function,
                              bo_config=config)
    bo.bayesian_optimization(init_budget=config.init_budget)
    bo.plot_bo_history()


if __name__ == '__main__':
    main_bo(config=bo_config)
