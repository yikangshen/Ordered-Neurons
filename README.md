# ON-LSTM

This repository contains the code used for propositional logic experiments in 
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
Python 3.6 and PyTorch 0.4 are required for the current codebase.

## Steps

1. Install PyTorch 0.4

2. Scripts and commands

  	+  Train & test logical inference model 
  	```python proplog.py --save logic.pt --cuda```