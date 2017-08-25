# InsuranceQA_zh
Question and Answering Language Model

# Welcome
Convolutional Neural Network for Chinese InsuranceQA Corpus with TensorFlow, implementation of paper ["Applying Deep Learning to Answer Selection: A Study and An Open Task"](https://arxiv.org/abs/1508.01585).

![](./assets/nn.png)

## Deps
* py2
* TensorFlow v1.0+

```
pip install --upgrade insuranceqa_data
pip install tensorflow-gpu==1.2
```

# Train
```
python2 train.py
```

Customize the hyper parameters: ```python2 train.py --help```

# Metrics
```
scripts/start_tensorboard.sh
open http://localhost:6006
```

> 在默认参数下，运行57,123 steps: loss 0.591968, acc 0.8

![](./assets/loss.png)

# Data
[insuranceQA Chinese Corpus](https://github.com/Samurais/insuranceqa-corpus-zh)

# Documentation
https://github.com/l11x0m7/InsuranceQA_zh/wiki

* 网络设计

* 数据设计

* 调优过程

* 结果

* 参考文献

# Others
[insuranceQA English Corpus](https://github.com/l11x0m7/InsuranceQA)

#  License
[Apache 2.0](./LICENSE)