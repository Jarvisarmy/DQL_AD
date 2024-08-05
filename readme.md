## Download Datasets
Please download MVtecAD dataset from [MVTecAD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/), Brats2021 dataset from [BraTS 2021 dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1), BTAD dataset from [BTAD dataset](https://github.com/pankajmishra000/VT-ADL) and IXI dataset from [IXI dataset](https://brain-development.org/ixi-dataset/).

## Training
For straight visualization on the training process of reinforcement. Curent ipynb files may contains abundant code that I use to visualize or print result. A more organized version may be updated if anyone is needed.
- experiment_brats.ipynb contains code for experiments on BraTS
- experiment_BTAD.ipynb contains code for experiments on BTAD
- experiment_mvtec.ipynb contains code for experiments on MVTec AD

## Pre-processing

 - [src/data_utils.py](data_utils.py) contains code for brain extraction, volume registered and converting volumes to slices. As mentioned in paper, we use [HD-BET](https://github.com/MIC-DKFZ/HD-BET) to skull-tripped and use [Flirt](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) to registered brain MRI volumes.

 - function [brain_extraction] is used to perform brain extraction using HD-BET
 - function [registered_nii_IXI] is used to registered the volumes from IXI to standard template
 - function [registered_nii_BraTS] is used to registered the volumes from BraTS to standard template
 - function [load_slices_for_3D] is used convert each volume to slices and stored in folder BraTS2D which is used to trained 2D methods.
 - for BTAD dataset, we change the folder name of normal images from 'ok' to 'good' to match the format of MVTec AD.

## Information
- please contact me if you need my pre-processed brain MRI slices or checkpoints to replicate my paper's result.
