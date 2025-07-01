import time

import wandb
from bo.bo import BayesianOptimization
from helper import Options, device, set_seed, task_config
from resnet.resnet import ResNet, create_data_loader, train_restnet_with_lr, visualize_classification_results


def main(config: Options):
    '''Main function to run the Bayesian Optimization for ResNet tuning.

    Args:
        config (Options): Configuration with hyperparameters for BO and ResNet.
        For details, see the docstring in helper.py -> Options.

    NOTE: The learning rate search space is set to be between 1e-5 and 1.0, where
    applying a log10 transformation makes more sense. Therefore,
    the log10 transformation is applied as in helper.py -> task_config (Options)
    '''
    # Initialize wandb
    wandb.init(
        project='bo-resnet',
        # name=f"bo_run_{config.seed}",
        config={
            'seed': config.seed,
            'resnet_num_epochs': config.resnet_num_epochs,
            'minibatch_size': config.minibatch_size,
            'gp_num_epochs': config.gp_num_epochs,
            'gp_learning_rate': config.gp_learning_rate,
            'kernel': config.kernel,
            'budget': config.budget,
            'init_budget': config.init_budget,
            'acquisition_optimization_budget': config.acquisitiion_optimization_budget,
            'hp_search_space': config.hp_search_space.tolist(),
            'hp_search_space_transformation': config.hp_search_space_transformation,
            'device': str(device)
        }
    )

    # set the random seed for reproducibility
    set_seed(config.seed, device=device)

    # Fashion-MNIST dataset loading
    train_loader, test_loader = create_data_loader(
        batch_size=config.minibatch_size
    )

    # create the BO instance
    bo = BayesianOptimization(
        unknown_function=lambda x:
        train_restnet_with_lr(train_loader,
                              test_loader,
                              num_epoch=config.resnet_num_epochs,
                              learning_rate=x),
        bo_config=config
    )

    # perform Bayesian Optimization
    bo_start_time = time.time()
    bo.bayesian_optimization(init_budget=config.init_budget)
    bo_end_time = time.time()
    bo_duration = bo_end_time - bo_start_time
    print(f'BO took {bo_duration:.2f} seconds')

    # Log BO duration to wandb
    wandb.log({'bo_duration_seconds': bo_duration})

    # plot the BO history
    bo.plot_bo_history()

    best_idx, best_lr, best_accuracy = bo.get_incumbent()

    # Log best results to wandb
    wandb.log({
        'best_learning_rate': best_lr,
        'best_test_accuracy': best_accuracy,
        'best_iteration': best_idx,
        'total_iterations': len(bo.sampled_x)
    })

    print(f'Best learning rate: {best_lr:.6f}')
    print(f'Best test accuracy: {best_accuracy:.2f}%')

    model = ResNet().to(device)
    model.load_state_dict(bo.model_history[best_idx])

    # OPTIONAL: re-train the ResNet with the best learning rate
    # print(f're-training ResNet with best learning rate: {best_lr:.5f}')
    # best_accuracy = train_restnet_with_lr(train_loader, test_loader, model,
    #                                            num_epoch=config.resnet_num_epochs,
    #                                            learning_rate=best_lr)

    # visualize the training results of ResNet
    final_metric = {
        'test_accuracy': best_accuracy,
    }
    visualize_classification_results(model, test_loader,
                                     metric=final_metric,
                                     save_dir=config.results_dir)
    model.save_model(
        save_dir=config.results_dir,
        filename='resnet_final.pth'
    )

    # Log plots to wandb
    try:
        # Log BO history plot
        bo_history_path = config.results_dir / 'bo_history.png'
        if bo_history_path.exists():
            wandb.log({'bo_history': wandb.Image(str(bo_history_path))})

        # Log classification results
        classification_results_path = config.results_dir / 'classification_results.png'
        if classification_results_path.exists():
            wandb.log({'classification_results': wandb.Image(
                str(classification_results_path))})

        # Log all BO iteration plots
        for i in range(len(bo.sampled_x)):
            iteration_plot_path = config.results_dir / \
                f'bo_iterations_results_{i+1}.png'
            if iteration_plot_path.exists():
                wandb.log(
                    {f'bo_iteration_{i+1}': wandb.Image(str(iteration_plot_path))})

    except Exception as e:
        print(f'Warning: Failed to log some plots to wandb: {e}')

    # Log final metrics
    wandb.log({
        'final_test_accuracy': best_accuracy,
        'final_learning_rate': best_lr
    })

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    # train with log10 transformation for learning rate
    print('Running Bayesian Optimization for ResNet with log10 '
          'transformation for learning rate...')
    main(config=task_config)

    # Comment out the following to test without the log10 transformation
    # or check the results in the saved_results_no_transform directory
    # import pathlib
    # from copy import deepcopy
    # print('Running Bayesian Optimization for ResNet without'
    #       ' log10 transformation for learning rate...')
    # no_transform_task_config = deepcopy(task_config)
    # no_transform_task_config.hp_search_space_transformation = None
    # no_transform_task_config.results_dir = pathlib.Path('results_no_transform')
    # main(config=no_transform_task_config)
