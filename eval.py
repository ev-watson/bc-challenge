import torch
import numpy as np

import config
from model import BinaryClassifier
from mlb_requests import Server

print("LOADING MODEL...")
model = BinaryClassifier()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

print("GETTING ATTEMPT...")
s = Server()
s.get()
binary_input = torch.tensor(np.frombuffer(s.binary, dtype=np.uint8), dtype=torch.float32)

print("MAKING GUESS...")
guess = model(binary_input.unsqueeze(0)).argmax(dim=1).item()
print(f"Model guess: {config.ARCHITECTURES[guess]}")
s.post(config.ARCHITECTURES[guess])

s.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(config.ARCHITECTURES[guess], s.ans, s.wins))