## COMP 551: Applied Machine Learning
### P4 - Reproducibility in ML

---
Fall 2021 @ McGill University <br>
Group 19 <br>
Paper: https://arxiv.org/pdf/2004.07320v3.pdf
---

#### Please refer to the following Colab notebooks for in-depth steps for reproducing our experiments: <br>

Paper Results Reproduction: <br>
https://github.com/felixsimard/comp551-p4/blob/main/results_reproduction.ipynb

MNIST Custom MLP Quantization Experiment: <br>
https://github.com/felixsimard/comp551-p4/blob/main/MNIST_Quantization_experiments.ipynb

Quantization Aware Training Experiment: <br>
https://github.com/felixsimard/comp551-p4/blob/main/Felix_QAT.ipynb

---

## Setup

### Prerequisite:

Using Python: `Python 3.8.7`

Make sure pip is installed: `pip --version` <br>

### Fairseq packages
To download the fairseq library along with its modules, use the following commands: <br><br>
`pip install regex requests hydra-core omegaconf bitarray` <br><br>
`!pip install fairseq` <br>

Note, you will most likely need to restart your Google Colab runtime once the above packages are downloaded.

### Additional packages
```
numpy==1.21.4
Pillow==8.4.0
torch==1.10.0
torchvision==0.11.1
typing-extensions==4.0.1
```
---

## Experiments

For the attempted reproduction of the paper results, make sure to execute the following commands in your notebook: <br>

`pip install tensorboardX` <br>

`pip install regex requests hydra-core omegaconf bitarray` <br>

`!pip install fairseq` <br>

````
git clone https://github.com/NVIDIA/apex
%cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
%cd ..
````

````
%cd fairseq
./examples/roberta/preprocess_GLUE_tasks.sh /content/drive/MyDrive/glue_data RTE
%cd ..
````

For the other experiments, these are the imports to our notebooks so ensure all imports are resolved correctly. <br>

````
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import torch.nn.functional as F
import numpy as np

from fairseq.modules.quantization.pq import quantize_model_, SizeTracker
from fairseq.modules.quant_noise import quant_noise

from operator import attrgetter, itemgetter
import re
````
