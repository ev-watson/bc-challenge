# Binary Architecture Classifier ML Task for Praetorian Challenge (https://www.praetorian.com/challenges/machine-learning-binaries/)

My attempt at it, no agentic assistance (no LLM generated code, and no commands were run by an agent at all).

Uses embedding to achieve behavioral/functional information encoding\
Then pooling for statistical aggregation (effectively acting as an inductive bias)\
Then residue pooling over the embedded vectors to achieve relatively cheap periodic positional encoding\
Then passes into a standard 5 layer MLP with hidden_dim=256 with standard generalization features (dropout, weight decay, layer norms)

Has about ~98% accuracy and does not use the provided possible target list as a prior

Model is available at `bcmodel.pt`, usage:
```
import torch
import numpy as np

import config
from model import BinaryClassifier

# load model and turn off weight adjustment
model = BinaryClassifier()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

# load input
BINARY = ... # Some np.ndarray of len 64 dtype=np.uint8
binary_input = torch.tensor(BINARY, dtype=torch.long)

# make output and take argmax to isolate highest probability
output = model(binary_input.unsqueeze(0)).argmax(dim=1).item()
architecture = config.ARCHITECTURES[output] 
```

