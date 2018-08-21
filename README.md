# **BTCWavelet:**
### An attempt to predict Bitcoin bubble crashes.

* Implementation of Log-Periodic Power Law fit on Bitcoin hourly data series
* Crawling hourly Bitcoin data Using [Cryptocompare](https://min-api.cryptocompare.com/) Historical Data API
* Reconstructing Bitcoin Time series using Discrete Wavelet Transform using [PyWavelets](https://pywavelets.readthedocs.io/en/latest/#)
* Implementation of the [Filimonov & Sornette](https://arxiv.org/abs/1108.0099v3) Epsilon Drawdown algorithm
* Some Genetic Algorithm attempt using [SAWADA Takahiro](https://github.com/fanannan/LPPL) code

### Content:
[1. Creating Anaconda Environment:](#1) 

[2. Running the code:](#2) 


### 1. Creating Anaconda Environment: <a  id="1"></a> 

The easiest way to replicate the Python environment needed to run this code is to use the file conda-spec.txt to create an identical local conda environment by invoking the following command from the repository directory:

```conda create --name tensorflow --file conda-spec.txt ```

For more information on how conda environments work refer to [Conda user guide: Building Identical Conda Environments](https://conda.io/docs/user-guide/tasks/manage-environments.html#building-identical-conda-environments) 

### 2. Running the code: <a  id="2"></a>  

All the experiments starting points are in [lppl_ga.py](../blob/master/code/lppl_ga.py). The code will default to run a 1000 Bitcoin trials on hourly data with DWT enabled. I might create a command-line interface for the different experiments in the future. 
