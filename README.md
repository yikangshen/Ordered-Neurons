# ON-LSTM

This repository contains the code used for word-level language model and unsupervised parsing experiments in 
[Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536) paper, 
originally forked from the 
[LSTM and QRNN Language Model Toolkit for PyTorch](https://github.com/salesforce/awd-lstm-lm).
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

2. Download PTB data. Note that the two tasks, i.e., language modeling and unsupervised parsing share the same model 
strucutre but require different formats of the PTB data. For language modeling we need the standard 10,000 word 
[Penn Treebank corpus](https://github.com/pytorch/examples/tree/75e435f98ab7aaa7f82632d4e633e8e03070e8ac/word_language_model/data/penn) data, 
and for parsing we need [Penn Treebank Parsed](https://catalog.ldc.upenn.edu/LDC99T42) data.

3. Scripts and commands

  	+  Train Language Modeling
  	```python main.py --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --data /path/to/your/data```

  	+ Test Unsupervised Parsing
    ```python test_phrase_grammar.py --cuda```
    
    The default setting in `main.py` achieves a perplexity of approximately `56.17` on PTB test set 
    and unlabeled F1 of approximately `47.7` on WSJ test set.
