
# A computational analysis of flow dynamics and heat transfer in a wavy patterned channel using PINNs

## Overview
Trying to replicate the paper mentioned above. No code or dataset has been provided by authors, so it's just hit & try which we have to follow to replicate, as many of the hyperparams has not been fully explained by the authors.

## Setup
- Have a `python` or the `conda` env or create one
- For `python` env, run the command `pip install Scripts/reqs.txt`
- For `conda` env, run the command `conda env create -f environment.yml`

This will install the required packages in your envs and then you are good to go for the replication.
Even if we have any issue then install the below mentioned packages(minimal):

```bash
pip install numpy pandas pytorch matplot scipy
```


## Folders and Files

## Folder: testonly/adaptive_beta
- `gradnorm`: Train the symmetric on using the adaptive beta following the gradient based beta optimization(try to balance the gradient of the both losses). Params for the alpha is `0.05`. *Code is commented in the train.py*.
- `gradnorm_001`: Same as the above but params for the alpha is `0.001`.
- `rationNorm`: On symmetric channel with ratio based optimization (tries to maintain the ratio of both loss) with params `beta_update_alpha` as `0.05`. *Code is commented in the train.py*.
- `ratioNorm_001`: Same as the above but with params as `0.001`.


### Folder: eval_data

- Holds the 8 csv files as `cfd_30.csv`, `cfd_60.csv`, `cfd_90.csv`, `cfd_180.csv`, `cfd_m30.csv`, `cfd_m60.csv`, `cfd_m90.csv`, `cfd_sym.csv`.
- In above files `cfd` denotes the computational fluid dynamics and they are in the name denoting this files have been generated via cfd whose model is in the folder `CFD-Model` and text after the `_` denotes the angle used in geometry of that simulation means `60` means we have +60 degree in phase and `m90` means we were 90 degree out phase in that simulation.
- `sym` refers to the symmetric diagram as shown in the paper which is neither in phase not out phase.
- Each files have 5 cols as `x,y,u,v,p` denoting the x-coords, y-coords, velocity in x direction, velocity in y direction and pressure respectively.
- We have same `x,y` for each files as evaluated on the same mesh and domain.
- Each of the above files have `12091` data points where we will be evaluating our model performance.


### Folder: others

- `Explained.md`: File trying to explain the data collected
- `geometry.png`: File holding the 8 geometry structure which we are working on
- `info.md`: key datas extracted from the paper
- `paper.pdf`: Work which we are replicating

### Folder: CFD-model

- `model.mph`: COMSOL-6.2 CFD model used to replicate the flow used in paper


### Folder: points
This folder holds the points which we are using now to train our PINNs as they are closest to the paper's distribution. Points are distributed via the `LHS` but refinement has been done manually as no info regarding it has been provided.

- Holds the 8 folders as `cfd_30/`, `cfd_60/`, `cfd_90/`, `cfd_180/`, `cfd_m30/`, `cfd_m60/`, `cfd_m90/`, `cfd_sym/`
- Each folder have 3 files as `boundary.csv`, `collocation.csv` and `collocation.png`
- `boundary.csv`: File with 8k/9k points alloted on the boundary of the domain
  - Each entry in `boundary` have `x, y, nx, ny, type` representing the x coords, y coords, x normal, y normal and type of point
  - Above type refers to the `upper wall`, `lower wall`, `inlet`, `outlet` which are self explaining
- `collocation.csv`: File for the collocation points means the inner points of the geometry where we will be imposing the PDEs.
  - Have 89.6k/62.7k points on geometry
  - Each entry consists of the `x, y, zone` where x, y represents the spatial coordinates and `zone` represents the region where our point lies
  - `core, bl, corner`: These are the 3 zones which we are using where core represents the interior points which are little away from the boundary, bl represents the points lying in the area closer to the boundary (we have defined certain boundary distance to consider it) and corner represents the points which lie on the corner of the boundary.
- `collocation.png`: Image showing the geometry and points which we are using


## Current Best
- `runs/sym_Re10_beta2_patience05` holds the best replication in `u,v` but we are trying to make it more robust so that we can do the same for the `p,T` too.
- `python eval1.py --checkpoint runs/sym_Re10_beta2_patience05/model_final.pt --ref_csv eval_data/cfd_sym.csv --out_dir runs/sym_Re10_beta2_patience05/eval`: cmd to evaluate the same
- `python train.py --geom_dir points/cfd_sym --Re 10 --Pr 0.7 --thermal none --out_dir runs/sym_Re10_beta2_patience05 --batch_size 8192 --beta 2.0`: cmd to train the same
- `python train_multi.py`: cmd to train for the multiple geometries.

## Reference to the pressure's loss terms
- `https://share.gemini.google/T4jHYQZAHLiK`

## Authors
- *rajm012*
- *rajmyra*
