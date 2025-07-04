# BO_ResNet

## Overview
Find the optimal learning rate of a ResNet using Bayesian Optimization (BO)

## Project Structure

```
BO_ResNet/
├── README.md
├── requirements.txt           # Python dependency
├── pixi.toml                  # Pixi project management
├── Dockerfile                 # Docker image configuration
├── .dockerignore
├── bo/                        # Bayesian Optimization implementation
│   ├── __init__.py
│   └── bo.py                  # BO class with GP and acquisition functions
├── resnet/                    # ResNet model and training utilities
│   ├── __init__.py
│   └── resnet.py              # ResNet architecture and training function
├── helper.py                  # Configuration classes and helper functions
└── main.py                    # Main script to run experiment
```

### Clone repo
```bash
git clone git@github.com:MingxuanChe/BO_ResNet.git
cd BO_ResNet
```

#### Option 1: Create a `conda` environment
Create and access a Python 3.10 environment using `conda` [[link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)]


```bash
conda create -n bo-resnet python=3.10
conda activate bo-resnet
```
Install dependency
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

#### Option 2: Using `pixi` environment
Download `pixi` [[link](https://pixi.sh/dev/installation/)]

Initialize shell with
```bash
pixi shell
```
**NOTE**: You might need to deactivate `conda` env

#### Option 3: Using Docker
Build the Docker image:
```bash
docker build -t bo-resnet .
```

Run the container:
```bash
# Run with volume mounting
# On Linux
docker run --rm -v $(pwd)/results:/project/results bo-resnet

# On Windows
docker run --rm -v %cd%/results:/project/results bo-resnet
```

## Usage
Run Bayesian Optimization (BO) for ResNet learning rate tuning:

```bash
python3 main.py
```
This main script will:
1. Load configurations, including a pre-defined learning rate search space `[1e-5, 1.0]`
2. Initilaize the BO with a Sobol sequence
3. Run BO to find the optimal learning rate
4. Save BO results and visualization to `./results`

Examplary results can be found in `./saved_results`, which contains the following
- `bo_iteration_results_{iteration_index}.png`
  - Upper subplot: all observations, the posterior mean, 3-sigma confidence interval
  - Bottom subplot: acquisition function value over search space and the candidate to be sampled next
  - **NOTE**: when a search space transformation is performed, the bottom axis of each subplot shows the transformed search space. An additional axis is at the top margin of the subplot for the original search space
- `bo_hisotry.png`
  - Upper subplot: fitted GP, observations etc at the end of BO iterations
  - Bottom subplot: BO history
- `classification_results.png`
  - Exemplary output of the ResNet trained with the optimal learning rate

## Additional note
- I like `pixi` and I recommend it
- ResNet has three building blocks, implemented following the spirit of the original paper
- GP and BO are implemented with `gpytorch`, which is also the main GP library used for my current projects
- PEP8 is checked with `pre-commit`
- Hyperparameters for both GP adn ResNet are not tuned heavily. For BO, I picked a set of hyperparameters with which GP doesn't show significant overfitting at the end of BO. Since the total budget is small in this task, specifying a prior/initial guess on the kernel hyperparameters, e.g., length_scale and output_scale, might make sense (omitted for simplicity).
- Acquisition function maximization is done with scipy `minimize`, following [BoTorch doc](https://botorch.org/docs/optimization#using-scipy-optimizers-on-tensors). However, we can observe that it is not always find the maximum. More sophisticated solver is not implemented for simplicity.
- From my experience, neural nets usually works good with a learning rate smaller than 1. Therefore, the search space is chosen to be `[1e-5, 1.0]`. For this search space, my intuition is that a `log10` transformation of the search space should make BO converge faster to a better optimum. The comparison can be found in `saved_results_no_transform` (without transformation) and `saved_results` (with transformation). The final test accuracy are 89.99% and 90.60% for without transformation and with transformation, respectively. In this test, with transformation converges towards optimum faster.
- excluded `wandb` for simplicity
