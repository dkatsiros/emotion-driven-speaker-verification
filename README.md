# Emotion Driven Speaker Verification



#### This repository contains code for my diploma thesis with supervisors [Theodore Giannakopoulos](mailto://tyiannak@gmail.com) and [Alexandros Potamianos](mailto:potam@central.ntua.gr).

## Abstruct
Speaker Verification (SV) enables the authentication of a claimed identity from measurements on a voice signal. Emotion as a natural and often involuntary encoder of voice, has the mechanisms responsible for vocal modulation. Despite the attention that the field
has gained over the years, little effort has been made in order to identify the relations between these two subjects. Αlthough seemingly far, emotional content could have a huge impact on speaker discrimination.

In this thesis, we investigate the correlation between speaker verification and speech emotion recognition. First of all, we create various emotional evaluation sets, each one aiming to track differently the effect of emotion on the speaker verification task. In an
attempt to decrease or even eliminate the effect we try to transfer emotional knowledge to our task. For this purpose, we implement four different architectures, each one of them, handling emotional information in a different manner. Then we examine our models’ performance on the emotional evaluation sets.

Our results suggest that emotional information has a crucial role on speaker verification. Even on low intensity, emotion on both on enrollment and verification can significantly degrade a system’s performance. On addition, emotions on strong intensity, seem to escalate the effect and ensue in poor results. Among the seven emotions examined, we find that, anger and fear were these having the most remarkable impact.

In an endeavor to address the aforementioned issues we examine the performance of our emotion-aware architectures. Our results indicate that by applying classic fine tuning techniques, we are able provide emotion robust models and at the same time perform much
better on the speaker verification task. Last but not least, we test our hypothesis on providing same-emotion utterances on evaluation phase and we observe a relative improvement around 20%, irrespective of emotional pre-training.

Overall, we can capture a strong relation between speaker discrimination and emotional content. We contend that controlling emotional content is necessary for a model’s
robustness, especially for real life scenarios, where emotion is present. Ultimately, we can
reduce the effect and improve our models performance by applying traditional transfer
learning techniques from speech emotion recognition to speaker verification.

## Repository Structure
- core:
The main scripts for running each experiment on each dataset.
- dataloading:
Custom PyTorch DataLoaders for loading each dataset (emodb, iemocap, ravdess, timit and voxceleb).
- lib:
All the necessary libraries such training procedures, model editing functions, sound processing tools, metrics for model evaluations and loss functions.
- models:
PyTorch implementations of models.
- scripts: Python and Bash scripts for repository manipulation and minor functions.
- utils: Many utilities for dataset utilization (such as emotional descriptors etc based on each datasets' description), loaders and early stopping implementation.