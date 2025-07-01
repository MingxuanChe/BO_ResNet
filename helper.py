
import pathlib

import numpy as np
import torch

# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Options(object):
    '''
    A class for easy access to hyperparameters.
    Configuration for the ResNet training task with Bayesian Optimization (BO).

    Details for this task:
        - General:
            seed (int): Random seed for reproducibility.
            results_dir (pathlib.Path): Directory to save the results.
        - ResNet Hyperparameters:
            resnet_num_epochs (int): Number of epochs for training the ResNet model.
            minibatch_size (int): Size of the minibatch for training.
            resnet_learning_rate (float): Learning rate for the ResNet model.
        - Gaussian Process (GP) Parameters:
            gp_num_epochs (int): Number of epochs for training the Gaussian Process (GP).
            gp_learning_rate (float): Learning rate for the GP.
            kernel (str): Kernel type for the GP, either 'RBF' or 'Matern'.
            lengthscale_init_mean (float): Initial mean for the lengthscale parameter.
            outputscale_init_mean (float): Initial mean for the outputscale parameter.
            noise_init_var (float): Initial variance for the noise parameter.
            lengthscale_prior_mean (float): Prior mean for the lengthscale parameter.
            lengthscale_prior_std (float): Prior standard deviation for the lengthscale parameter.
            outputscale_prior_mean (float): Prior mean for the outputscale parameter.
            outputscale_prior_std (float): Prior standard deviation for the outputscale parameter.
            noise_prior_mean (float): Prior mean for the noise parameter.
            noise_prior_std (float): Prior standard deviation for the noise parameter.
        - Bayesian Optimization (BO) Parameters:
            budget (int): Budget for Bayesian Optimization (BO).
            init_budget (int): Initial budget for the Sobol sequence initialization.
            acquisition_optimization_budget (int): Budget for optimizing the acquisition function.
            hp_search_space (np.array): Search space for the hyperparameter, defined as a 2D numpy array.
            hp_search_space_transformation (str or None): Transformation for the
    '''

    def __init__(self, **kwargs):
        super(Options, self).__init__()
        self.__dict__.update(kwargs)


def set_seed(seed, device=torch.device('cpu')):
    # set the random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)


# for ResNet training
resnet_config = Options(
    # the original paper uses SGD; we use Adam instead
    # general
    seed=1,
    results_dir=pathlib.Path('results'),
    # ResNet hyperparameters
    resnet_num_epochs=10,
    minibatch_size=256,
    resnet_learning_rate=0.01,
)


# for Bayesian Optimization (BO) sanity check
bo_config = Options(
    # general
    seed=1,  # random seed for reproducibility
    results_dir=pathlib.Path('results'),
    # GP parameters
    gp_num_epochs=1000,
    gp_learning_rate=0.005,
    # kernel='RBF',
    kernel='Matern',
    lengthscale_init_mean=1.0,
    outputscale_init_mean=5,
    noise_init_var=1e-4,
    # BO parameters
    bo_init_method='sobol',
    budget=10,
    init_budget=3,
    acquisitiion_optimization_budget=30,
    hp_search_space=np.array([[0.001, 4.0]]),
    hp_search_space_transformation=None,
)

# for the ResNet training task with Bayesian Optimization (BO)
task_config = Options(
    seed=1,
    results_dir=pathlib.Path('results'),
    # ResNet hyperparameters
    resnet_num_epochs=10,
    minibatch_size=256,
    resnet_learning_rate=0.01,
    # GP parameters
    gp_num_epochs=1000,
    gp_learning_rate=0.005,
    # kernel='RBF',
    kernel='Matern',
    lengthscale_init_mean=1.0,
    outputscale_init_mean=70,
    noise_init_var=0.5,
    # lengthscale_prior_mean=1.0,
    # lengthscale_prior_std=2.0,
    outputscale_prior_mean=70,
    outputscale_prior_std=20,
    # noise_prior_mean=0.1,
    # noise_prior_std=1.0,
    # Bayesian Optimization parameters
    budget=10,
    init_budget=3,
    bo_init_method='sobol',
    acquisitiion_optimization_budget=200,
    # kernel='Matern',
    hp_search_space=np.array([[0.00001, 1.0]]),
    hp_search_space_transformation='log10',  # log transformation makes sense
    # hp_search_space_transformation=None,  # no transformation for the search space
)
