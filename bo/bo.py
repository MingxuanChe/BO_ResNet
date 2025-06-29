import numpy as np
import torch
import gpytorch
import pathlib
from gpytorch.kernels import ScaleKernel, RBFKernel
from scipy.stats import qmc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# set seaborn style
sns.set_theme(style='darkgrid', palette='deep')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Options(object):
    def __init__(self, **kwargs):
        super(Options, self).__init__()
        self.__dict__.update(kwargs)
        
BO_config = Options(
    seed=1,  # random seed for reproducibility
    # the original paper uses SGD; we use Adam instead
    # training HPs
    num_epochs=500,
    default_learning_rate=0.1,
    kernel='RBF',  # 'RBF' or 'Matern'
    # kernel='Matern', 
    results_dir=pathlib.Path('results'),
    acquisitiion_optimization_budget=20,  # budget for acquisition function optimization
    hp_search_space=np.array([[0.00001, 1.0]]) # log transformation makes sense
)

# set the random seed for reproducibility
np.random.seed(BO_config.seed)
torch.manual_seed(BO_config.seed)
if device == torch.device("cuda"):
    torch.cuda.manual_seed(BO_config.seed) 
    

'''
we let BO consider maximization problem
'''

def get_sobol_init(num_sample, num_dim=1, bounds=None):
    """
    Generate Sobol sequence samples in the specified bounds
    """
    sobol = qmc.Sobol(d=num_dim)
    samples = sobol.random(n=num_sample, workers=2)
    # will ample in [0, 1) by default
    if bounds is not None:
        low, high = bounds[:, 0], bounds[:, 1]
        samples = low + (high - low) * samples
        assert np.all(samples >= low) and np.all(samples < high), \
        "Samples are not within the specified range."
    return samples

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, 
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                 mean_module=gpytorch.means.ConstantMean(),
                 kernel='RBF',
                 ):
        """
        Initialize the ExactGP model with specified kernel
        """
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        if kernel == 'RBF':
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel == 'Matern':
            self.covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        
    def forward(self, x):
        """
        Forward pass
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class BayesianOptimization:
    def __init__(self, 
                 unknown_function, 
                 search_space=None,
                 budget=10, kernel='RBF',
                 ):
        """
        Initialize the Bayesian Optimization model
        """
        self.unknown_function = unknown_function
        self.search_space = search_space if search_space is not None else np.array([[0.0, 1.0]])
            
        self.budget = budget
        self.remaining_budget = budget
        self.kernel = kernel

        # sample history        
        self.sampled_x = []
        self.sampled_y = []
                
    def initialize_bayesian_optimization(self, method='sobol', init_budget=3):
        if method == 'sobol' and init_budget > 0:
            sobol_samples = get_sobol_init(num_sample=init_budget, 
                                           bounds=self.search_space).reshape(-1).tolist()
            for sample in sobol_samples:
                y_value = self.unknown_function(sample)
                self.sampled_x.append(sample)
                self.sampled_y.append(y_value)

            self.train_x = torch.tensor(self.sampled_x, dtype=torch.float32).reshape(-1)
            self.train_y = torch.tensor(self.sampled_y, dtype=torch.float32).reshape(-1)
            
            self.remaining_budget = self.budget - init_budget
        else:
            raise ValueError(f"Unknown initialization method: {method}")    

        # find the current incumbent
        if len(self.sampled_y) > 0:
            self.incumbent = torch.max(torch.tensor(self.sampled_y)).item()
            print(f"Initial incumbent value: {self.incumbent}")

        # subtract the budget
        self.remaining_budget = self.budget - init_budget
        
    def fit(self):
        """
        Fit the model to the training data
        """
        self.gp_model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.gp_model.parameters(), 
                                     lr=BO_config.default_learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        
        for _ in range(BO_config.num_epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.train_x)
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            optimizer.step()
            if _ % (BO_config.num_epochs/5) == 0:
                print(f'Epoch {_}: GP Loss = {loss.item()}')

    
    def expected_improvment(self, mean, std, incumbent, xi=0.0):
        '''
        Calculate the expected improvement (EI) given the mean, std, and incumbent value.
        NOTE: in the online lecture, minimization is used; in the BO tutorial paper + Murphy's book, maximization is used.
        '''
        # avoid division by zero
        if torch.isclose(std, torch.tensor(0.0)):
            return 0.0
        
        z = (mean - incumbent - xi) / std
        exploitation_term = (mean - incumbent - xi) * torch.distributions.Normal(0, 1).cdf(z)
        exploration_term = std * torch.distributions.Normal(0, 1).log_prob(z).exp()
        # exploration_term = std * norm.pdf(z.item())
        # exploration_term = torch.tensor(exploration_term, dtype=torch.float32)
        
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
        # use scipy.optimize.minimize (similary to BoTorch)
        best_x = None
        best_acquisition_value = -np.inf
        
        for _ in range(BO_config.acquisitiion_optimization_budget):
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
            raise ValueError("No valid acquisition point found.")
        
        return best_x, best_acquisition_value

    def bayesian_optimization(self, init_budget=3):
        """
        Perform Bayesian Optimization
        """
        # 1. initialize the dataset
        self.initialize_bayesian_optimization(method='sobol', init_budget=init_budget)
        # 2. loop until budget
        for i in range(init_budget, self.budget):
            # GP
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_model = ExactGP(self.train_x, self.train_y, 
                                    self.likelihood, kernel=self.kernel)
            
            # 3. fit the GP model
            self.fit()
            # 4. optimize the acquisition function
            next_x, _ = self.optimize_acquisition(self.acquisition_value_at)
            
            self.plot_iteration_results(i, next_x)
            # 5. query the unknown function
            next_y = self.unknown_function(next_x)
            # 6. update the training data
            self.sampled_x.append(next_x)
            self.sampled_y.append(next_y)
            
            self.train_x = torch.tensor(self.sampled_x, dtype=torch.float32)
            self.train_y = torch.tensor(self.sampled_y, dtype=torch.float32)
        
            print(f"Iteration {i + 1}: Next x = {next_x}, Next y = {next_y}")
            self.incumbent = torch.max(self.train_y).item()
            print(f"Current incumbent value: {self.incumbent}")
            
        # return the best found point
        incumbent_index = torch.argmax(self.train_y).item()
        best_x = self.train_x[incumbent_index].item()
        best_y = self.train_y[incumbent_index].item()  
        print(f"Best x: {best_x}, Best y: {best_y}")
        return best_x, best_y
    
    def plot_iteration_results(self, i, next_x, s=3):
        """
        Plot the results of the Bayesian Optimization iterations
        """
        x = np.linspace(self.search_space[0, 0], self.search_space[0, 1], 100)
        # the true objective might be expensive to evaluate
        # y = [self.unknown_function(xi) for xi in x]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp_model.eval()
            self.likelihood.eval()
            x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            pred = self.likelihood(self.gp_model(x_tensor))
            # get the mean and std of the GP model
            mean = pred.mean.numpy()
            std = pred.stddev.numpy()
        
        fig, ax= plt.subplots(2, 1, figsize=(10, 8))
        # plot the objective function
        # ax[0].plot(x, y, label='Objective Function', color='r')
        ax[0].plot(x, mean, label='GP Mean', color='blue')
        ax[0].fill_between(x, mean - s * std, mean + s * std, 
                           color='lightblue', alpha=0.5, label=f'{s}-$\sigma$ Confidence Interval')
        ax[0].scatter(self.sampled_x, self.sampled_y, color='k', label='Sampled Points')
        ax[0].axvline(x=next_x, color='orange', linestyle='--', label='Next Sample Point')
        ax[0].set_title('Objective Function with Sampled Points')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('f(x)')
        ax[0].legend()
        ax[0].grid()
        
        # plot the acquisition function
        acquisition_values = [self.acquisition_value_at(torch.tensor([[xi]], dtype=torch.float32)).item() for xi in x]
        ax[1].plot(x, acquisition_values, label='Acquisition Function', color='green')
        ax[1].scatter(next_x, self.acquisition_value_at(torch.tensor([[next_x]], dtype=torch.float32)).item(), 
                      color='orange', label='Next Sample Point', zorder=5)
        ax[1].axvline(x=next_x, color='orange', linestyle='--', label='Next Sample Point')
        ax[1].set_title('Acquisition Function with Sampled Points')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('Acquisition Value')
        ax[1].legend()
        ax[1].grid()
        
        
        fig.suptitle(f'Bayesian Optimization Iteration {i + 1}', fontsize=16)
        fig.tight_layout()
        
        # save the figure
        fig_path = BO_config.results_dir / f'bo_iterations_results_{i + 1}.png'
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        
def simple_objectve_function(x):
    """
    An example objective function for testing with a shallow local maximum and a steep local minimum. 
    """
    return -(x - 0.5)**2 + 3 * np.sin(x * 3) + 0.5


def main():

    
    bo = BayesianOptimization(unknown_function=simple_objectve_function,
                             search_space=np.array([[0.0, 4.0]]),
                             budget=10, kernel=BO_config.kernel)
    bo.bayesian_optimization()
    
    # result = minimize(lambda x: -simple_objectve_function(x), 
    #                   x0=np.array([0.2]), 
    #                   bounds=[(0.0, 1.0)], 
    #                   method='L-BFGS-B')
    # print(f"Optimal x: {result.x[0]}, Optimal value: {-result.fun}")
    # # plot the objective function

    # x = np.linspace(0, 1, 100)
    # y = simple_objectve_function(x)
    # plt.plot(x, y, label='Objective Function')
    # plt.scatter(result.x, -result.fun, color='red', label='Optimal Point')
    # plt.title('Objective Function with Optimal Point')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.grid()
    # # save the figure
    # fig_path = pathlib.Path('bo_objective_function.png')
    # fig_path.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(fig_path)
    
    # # uniformly sample the search space
    # # check whether the log and exp transformation makes sense
    # num_dim = 1
    # num_sample = 10
    # search_space = np.array([[0.00001, 1.0]])
    # sampled_points = np.random.uniform(search_space[:, 0], 
    #                                    search_space[:, 1], 
    #                                    size=(num_sample, num_dim))
    # # sort the sampled points
    # sampled_points = np.sort(sampled_points, axis=0)
    
    # # sample log transformation
    # log_search_space = np.log(search_space)
    # log_sampled_points = np.random.uniform(log_search_space[:, 0],
    #                                        log_search_space[:, 1], 
    #                                        size=(num_sample, num_dim))
    # log_sampled_points = np.sort(log_sampled_points, axis=0)
    # original_sampled_points = np.exp(log_sampled_points)
    
    # print(f"Sampled points: {sampled_points}")
    # print(f"Log-transformed sampled points: {log_sampled_points}")
    # print(f"Original sampled points after exp transformation: {original_sampled_points}")
    # # use exponential transformation then sample again
    

if __name__ == '__main__':
    main()