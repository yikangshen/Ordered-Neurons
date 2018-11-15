# ON-LSTM

This repository contains the code used for targeted syntactic evaluation experiments in 
[Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536) paper, 
originally forked from the 
[Targeted Syntactic Evaluation of Language Models](https://github.com/BeckyMarvin/LM_syneval).
If you use this code or our results in your research, we'd appreciate if you cite our paper as following:

```
@article{shen2018ordered,
  title={Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks},
  author={Shen, Yikang and Tan, Shawn and Sordoni, Alessandro and Courville, Aaron},
  journal={arXiv preprint arXiv:1810.09536},
  year={2018}
}
```

## Software Requirements
Python 3.6, NLTK and PyTorch 0.4 are required for the current codebase.

## Steps

1. Install PyTorch 0.4 and NLTK

2. Download training data. We used the same training data as Gulordava et al. (2018). Each corpus consists of around 100M tokens from English Wikipedia. We used training (80M) and validation (10M) subsets in our experiments. All corpora were shuffled at sentence level. Links to download the data are below:

[train](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/train.txt) / [valid](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/valid.txt) / [test](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/test.txt) / [vocab](https://s3.amazonaws.com/colorless-green-rnns/training-data/English/vocab.txt)


3. Scripts and commands

  	+ Train Language Modeling
  	```sh train_LM.sh```

  	+ Test Model
    ```sh test_Syn.sh```
    
    + Analyze the results
    ```sh analyze_resutls.sh```
    
    Results can be found at `results/rnn/`