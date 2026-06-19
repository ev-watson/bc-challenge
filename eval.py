import torch
import numpy as np

import config
from model import BinaryClassifier
from mlb_requests import Server

model = BinaryClassifier()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

s = Server()

for _ in range(550):
    s.get()
    binary_input = torch.tensor(np.frombuffer(s.binary, dtype=np.uint8), dtype=torch.long)
    
    guess = model(binary_input.unsqueeze(0)).argmax(dim=1).item()
    s.post(config.ARCHITECTURES[guess])

    s.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(config.ARCHITECTURES[guess], s.ans, s.wins))

    if s.hash:
        s.log.info(f"You win! {s.hash}")
        break
    
    if s.wins >= 499:
          s.log.info("RAW /solve: %s", s.json)

    if s.wins >= 501:
        print("WINNING THRESHOLD REACHED, HASH UNDETECTED, STOPPING...")
        break