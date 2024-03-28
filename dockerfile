FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04

RUN pip install wandb matplotlib pandas numpy biopython tokenizers
