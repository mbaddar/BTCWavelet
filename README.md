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

[3. Code Files:](#3)

[3. Classes Description:](#3)


### 1. Creating Anaconda Environment: <a  id="1"></a> 

The easiest way to replicate the Python environment needed to run this code is to use the file conda-spec.txt to create an identical local conda environment by invoking the following command from the repository directory:

```conda create --name tensorflow --file conda-spec.txt ```

For more information on how conda environments work refer to [Conda user guide: Building Identical Conda Environments](https://conda.io/docs/user-guide/tasks/manage-environments.html#building-identical-conda-environments) 

### 2. Running the code: <a  id="2"></a>  

All the experiments starting points are in [lppl_ga.py](../master/code/lppl_ga.py). The code will default to run a 1000 Bitcoin trials on hourly data with DWT enabled. I might create a command-line interface for the different experiments in the future. 

### 3. Code Files: <a  id="3"></a>

| File Name  |  Contained Classes |
|---|---|
|Lppl_ga.py|Pipeline, Nonlinear_Fit|
|Epsilon.py|Data_Wrapper, Epsilon_Drawdown|
|Decomposition.py|Wavelet_Wrapper|
|Crawler.py|BaseCrawler, Crawler, … exchange specific crawlers were not used|

### 4. Classes Description: <a  id="4"></a>
|Class name|Description|
|---|---|
| BaseCrawler  |  Basic http, requesting API endpoint functions |
|Crawler | Cryptocompare crawler. The Bitcoin hourly data source.  |
| Data_Wrapper  | Reads local Bitcoin and other financial assets data and returns them on a dataframe with a standard column formatting |
|  Epsilon_Drawdown  | Implementation of (Gerlach, et al., 2018) Epsilon Drwadown algorithm  |
| Nonlinear_Fit   | Implementation of LPPL non-linear optimization. The main algorithm used across experiments is Basin Hopping.|
| Pipeline   | The model pipeline from crawling to bubble prediction. Incomplete  |
| Wavelet_Wrapper   |  Implementation of DWT reconstruction using PyWavelets6 library |
