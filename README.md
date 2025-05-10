# RAPID: Residual-Aware Propagation for Infection Dynamics

This code is for the paper *Scaling Epidemic Inference on Contact Networks: Theory and Algorithms*, which investigates the probabilistic spread of epidemics on directed graphs and derives the steady-state distribution for each individual in the network. Our goal is to demystify the spread process and **accelerate the entire inference process**. For simplicity, the epidemic spread is modeled using the SIR framework, with Monte Carlo simulations serving as ground truth.

## Environmental Setting

First, create and activate the virtual environment:

```bash
conda create -n epidemics python==3.10

source activate epidemics
```


Run the following to install dependencies:

```bash
pip install -r requirements.txt
```

## Arguments

Here is a table of the argument parameters for the SIR model simulation:

| Argument | Default | Description                                         |
|----------|---------|-----------------------------------------------------|
| `--graph_type` | `"powerlaw"` | Graph type                                          |
| `--nodes` | `10000` | Number of nodes in the graph                        |
| `--average_degree` | `5` | Average degree of the graph                         |
| `--beta` | `1/18` | Infection rate                                      |
| `--gamma` | `1/9` | Recovery rate                                       |
| `--tol` | `1e-3` | Tolerance for convergence                           |
| `--num_simulations` | `50` | Number of Monte Carlo simulations                   |
| `--mc_batch_size` | `20` | Batch size of Monte Carlo simulations               |
| `--initial_infected_ratio` | `0.01` | Ratio of initially infected nodes in the population |
| `--top_k_ratio` | `0.1` | Top-K ratio for overlap calculation                 |
| `--p` | `10` | Number of pre-heat steps in LocalPush-PDS           |
| `--data_dir` | `"./data"` | Directory to load real data                         |
| `--file_name` | `""` | File name of the real data                          |
| `--output_dir` | `"./output"` | Directory to save results                           |

### Notes:
- The choices for `--graph_type` are: `ba`, `er`, `powerlaw`, `smallworld`.
- The choices for `--file_name` are `soc-sign-bitcoinalpha.csv.gz`, `hiv-Trans`,  `email-Enron.txt.gz`, `email-EuAll.txt.gz`, `soc-sign-epinions.csv.gz`, `soc-pokec-relationships.txt.gz`. If you want to run synthetic data, leave `--file_name` **empty**. Before running real datasets, please download them from [SNAP](https://snap.stanford.edu/data/index.html) and put them in the `data` directory.


## Usage
If you want to simulate with default parameters, simply run 

```bash
python main.py
```

If you want to run real datasets, you can specify the `--file_name` argument. For example, to run the Bitcoin-Alpha dataset, run

```bash
python main.py --file_name=soc-sign-bitcoinalpha.csv.gz --initial_infected_ratio=0.01
```

To run synthetic data, you can specify the `--graph_type` argument. For example, to run the Erdos-Renyi graph, run

```bash
python main.py --graph_type=er --nodes=10000 --average_degree=5
```

## Expected Output
After running `main.py`, there will be a folder name after the timestamp in the `output` directory. The folder contains the following files:

1. `arg_settings.csv`: Configuration file for the simulation.
```
Argument,Value
threshold,0.5
graph_type,
nodes,3783
average_degree,6.393338620142744
beta,0.05555555555555555
gamma,0.1111111111111111
tol,0.001
num_simulations,50
mc_batch_size,5
initial_infected_ratio,0.01
top_k_ratio,0.1
omega,1.3
p,20
data_dir,./data
file_name,soc-sign-bitcoinalpha.csv.gz
output_dir,./output/nv
```

2. `mc50_summary.csv`: Metrics for the simulation.
```
Metric,Value
Mean S,1571.6667
Lower (2.5%),1429.6083
Upper (97.5%),1730.8250
MC-50 Runtime (s),24.5565 ± 0.2765
MC-50 Iterations,86.63 ± 0.69
```

3. `centralities.csv`: Performance for centralities as baseline.
```
,Degree,Eigenvector,PageRank,Betweenness,Closeness
Kendall Tau,0.7470 ± 0.0043,0.5176 ± 0.0040,0.0338 ± 0.0058,0.6221 ± 0.0024,0.5375 ± 0.0132
Top-K Overlap,0.7354 ± 0.0037,0.6367 ± 0.0213,0.0194 ± 0.0054,0.5961 ± 0.0066,0.5573 ± 0.0156
Precision,1.0000,1.0000,0.6082,1.0000,0.5595
Recall,0.0010,0.0129,0.9461,0.0005,1.0000
F1 Score,0.0019,0.0254,0.7404,0.0010,0.7175
```

4. `DMP_summary.csv`: Performance for PDS.
```
Runtime (s),Iterations,Kendall Tau,Top-K Overlap,Normalized MSE,Normalized MAE
3.8806 ± 0.0144,88.00 ± 0.00,0.8471 ± 0.0007,0.7663 ± 0.0045,0.0035 ± 0.0002,0.0436 ± 0.0011
```

The content of summaries of  `MC-5`, `MC-10`, `DMP(No_Cavity)` and `RAPID` are similar in format of `DMP_summary.csv`.