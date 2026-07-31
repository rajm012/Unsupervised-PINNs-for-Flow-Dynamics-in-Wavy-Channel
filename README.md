
# A computational analysis of flow dynamics and heat transfer in a wavy patterned channel using PINNs

## Overview
Trying to replicate the paper mentioned above. No code or dataset has been provided by authors so it's just hit & try which we have to follow to replicate as many of the hyperparams has not been fully explained by the authors.

## Setup
- Have a `python` or the `conda` env or create one
- For `python` env, run the command `pip install Scripts/reqs.txt`
- For `conda` env, run the command `conda env create -f environment.yml`

This will install the required packages in your envs and then you are good to go for the replication.
Even if we have any issue then install the below mentioned packages(minimal):

```bash
pip install numpy pandas pytorch matplot scipy
```



## DataSets

- **Collocation Points and Boundary Points**: Use the code in the `Scripts/dataUsedGen.py` to generate the same.
- **CFD**: Use the model `model/replicate.mph` to get the points.

## Current Best
- `runs/sym_Re10_beta2_patience2k` holds the best replication in `u,v` but we are trying to make it more robust so that we can do the same for the `p,T` too.
- `python eval1.py --checkpoint runs/sym_Re10_beta2_patience2k/model_final.pt --ref_csv data/eval/cfd_sym.csv --out_dir runs/sym_Re10_beta2_patience2k/eval`: cmd to evaluate the same
- `python train.py --geom_dir data/sym --Re 10 --Pr 0.7 --thermal none --out_dir runs/sym_Re10_beta2_patience2k --batch_size 8192 --beta 2.0`: cmd to train the same



## Authors
- *rajm012*
- *rajmyra*
