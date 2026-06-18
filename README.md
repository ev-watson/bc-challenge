# Binary Architecture Classifier ML Task for Praetorian Challenge (https://www.praetorian.com/challenges/machine-learning-binaries/)

My attempt at it, no agentic assistance (no LLM generated code, and no commands were run by an agent at all).

Uses embedding to achieve behavioral/functional information encoding\
Then pooling for statistical aggregation (effectively acting as an inductive bias)\
Then residue pooling over the embedded vectors to achieve relatively cheap periodic positional encoding\
Then passes into a standard 5 layer MLP with hidden_dim=256 with standard generalization features (dropout, weight decay, layer norms)

Has about ~97.8% accuracy

