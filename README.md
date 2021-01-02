# A Linear Time Algorithm Finding Approximation Minimum Impurity Partitions

A Python implementation for algorithms 1, 2, and 3 in *A Linear Time Algorithm Finding Approximation Minimum Impurity Partitions*.

## Overview
This repository contains the implementations of two algorithms

- 'nguyen_algo': a linear time algorithm that approximates the minimum impurity partitions that can found in [1](#bib1).
- 'dc': The Divisive Information Theoretic Clustering from [2](#bib2).


## Dependencies
This project requires:
- Python 3 (tested with Python 3.8.5)
- scikit-learn (tested with scikit-learn 0.21.3)
- Numpy (tested with Numpy 1.18.0)
- Pandas (tested with Pandas 1.1.2)
To install all dependencies, 

```
pip install -r requirements.txt
```

## Retrieving and preprocessing the data
### Retrieving data
We tested the algorithms on the 20 news group dataset (ng20), which can be retrieved from the scikit-learn library, and the RCV1-v2 dataset that can be found [here](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm).

We also utilized the data preprocessing process from [3](#bib3) that can be found [here](https://github.com/lmurtinho/RatioGreedyClustering/tree/master). The preprocessed `ng20` dataset is included in this repository, but it can also be retrieved and saved by running `make_ng20_data.py`. For the RCV1-v2 dataset, it must be downloaded first from:
- http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
- http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz
- http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz
- http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz
- http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz
- http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz

unzipped and put into the folder that the `make_rcv1_data.py` points to. 


## Running experiment
### Re-run presented experiment
If there is the need to re-run the experiment that was presented in the paper, run 
```
python rerun_experiment.py -filename [filename] 
```
where `-filename` is the file path to the data file without the file extension.  Following the convention, the data file should be comma separated (`.csv` file). 

The output will be put into `[filename]_entropies.csv`. We also included the results from the paper with this repository. 

For instance, `python rerun_experiment.py -filename ng20` will run all the experiments on the *Newsgroup20* dataset and the output will can be found at `ng20_entropies.csv`.

For more information, run `python rerun_experiment.py -- help`.

### Run the algorithms with new dataset
Run
```
python main.py -filename [filename] -K [num_of_clusters]
```
where `-filename` is the file path to the data file without the file extension, and `-K` is the desired number of clusters. For instance, 
`python main.py -filename ng20 -K 20` will run both algorithms in [1](#bib1) and [2](#bib2) on the *Newsgroup20* dataset with K = 20. 

For more information, run `python main.py -- help`.


## Reference
<a id="bib1">1</a>: Nguyen, Le, Nguyen. A Linear Time Algorithm Finding Approximation Minimum Impurity Partitions, 2020.

<a id="bib2">2</a>: Dhillon, Inderjit S., Mallela, Subramanyam, and Kumar, Rahul. A divisive information-theoretic feature clustering algorithm for text classification. *Journal of Machine Learning Research*, 3:1265-1287, 2003.

<a id="bib3">3</a>: Cicalese, F., Laber, E. & Murtinho, L.. (2019). New results on information theoretic clustering. Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:1242-1251