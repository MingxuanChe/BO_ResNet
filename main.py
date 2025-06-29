
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch

from helper import Options, set_seed
from bo.bo import BayesianOptimization
from resnet.resnet import ResNet, create_data_loader, resnet_config, \
                          train_restnet_with_lr, visualize_classification_results

BO_resnet_config = Options(
    seed=1,  # random seed for reproducibility
    # training HPs
    num_epochs=500,
    default_learning_rate=0.1,
    kernel='RBF',  # 'RBF' or 'Matern'
    # kernel='Matern', 
    results_dir=pathlib.Path('results'),
    acquisitiion_optimization_budget=20,  # budget for acquisition function optimization
    hp_search_space=np.array([[0.00001, 1.0]]) # log transformation makes sense
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(BO_resnet_config.seed, device=device)        

def main():
    train_loader, test_loader = create_data_loader(batch_size=resnet_config.minibatch_size)
    bo = BayesianOptimization(unknown_function=\
                              lambda x: train_restnet_with_lr(train_loader, 
                                                              test_loader, 
                                                              model=None, 
                                                              learning_rate=x),
                             search_space=BO_resnet_config.hp_search_space,
                             budget=10, kernel=BO_resnet_config.kernel)
    bo.bayesian_optimization()
    
    model = ResNet().to(device)    
    model.load_state_dict(torch.load('resnet_final.pth'))
    model.eval()
    # visualize the training results
    visualize_classification_results(model, test_loader)

if __name__ == '__main__':
    main()