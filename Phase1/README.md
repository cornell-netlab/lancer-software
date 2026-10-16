# Phase I Agents

# SIS-* Agents

`Phase1/SIS_Agents` includes our:
- SIS-A Agent: `./Agents/LANCER/LANCER_SIS_A_v1.py`
- SIS-C Agent: `./Agents/LANCER/LANCER_SIS_C_v1.py`
- SIS-A Agent: `./Agents/LANCER/LANCER_SIS_I_v1.py`

These agent files include the model that was used for evaluation.

We have three additional models that are more resilient to different tpr and fpr values under `./SIS_Agents/LANCER/models/updated-tpr-fpr-models`.

# CAGE 2 Agents

`./Agents/LANCER/models/CAGE2_Agents` includes model weights for agents that can play in CAGE 2 and similar environments. An agent can be created by using `CleanRLCheckpointAgent` from `./Agents/LANCER/CleanRLAgent.py`.

