# Batch-Norm

This repository contains experiments for understanding the effect of Batch Normalization on parameter trajectory and its interaction with feature noise.

## Pre-requisites
For runnning the code, install the library [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings)

## Details
* `Understanding_Batch_Normalization.pdf:` Contains experiment details and results along with some theoretical justifications.
* `hessian.py:` Contains the code for plotting Hessian of DNN. Downloaded from [DeepnetHessian](https://github.com/AnonymousNIPS2019/DeepnetHessian) and modified appropriately.
* `OLS.ipynb:` Results showing impact of BN on the trajectory of OLS
* `BN_high_lr.pynb:` Empirically demonstrates that gradient explosion occurs for deeper nets without BN at high lr, while the same is not true for BN networks. Theoretical justification provided in pdf.
* `robust_noise.py:` Code for training a BN/non-BN network on the noisy dataset and producing the required logs.
* `plot_robust_noise.ipynb:` Plots the robustness of BN vs non-BN networks against noise with increasing data points using the logs generated from the above py file
* `Other_normalizations.py:` Plots the hessian and distance from initialization for other normalizations apart from BN. Some results have been provided in the folder Other_normalizations.
* `Hessian_analysis.py:` Produces a detailed analysis, including the entire Hessian, plots of top eigenvalues, Hessian split into different components etc. for BN and non-BN networks. A sample of the results produced has been given in the folders Hess_analysis_BN and Hess_analysis_no_BN.
