# Introduction
Updating soon

# Environment setting
## 1. Set the env path
In file ".envPath", replace "/directory-to-current-folder" with the directory of this folder. For example, "/user/project/LCZ-Cross-City-Transfer-Learning"

## 2. Install conda (conda 4.10.1, where the codes are tested)
[Please refer to the anaconda documentation](https://docs.anaconda.com/anaconda/install/)

## 3. Create the conda env
```bash
conda env create -f environment.yml #Create env from yml file
conda activate LCZ_TRANSFER #activate the conda env
```
# File descriptions
## Data
The folder stores data for all experiments
## Experiments
This folder includes scripts that run the algorithms shown in the paper. To re-run our experiments, please checkout this folder.
## SRC
This folder contains python libs that support the experiments
## Prediction

## Experiments_results.xlsx
This excel shows all experiments results that we report in the paper







