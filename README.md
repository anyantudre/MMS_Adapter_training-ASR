# Fine-Tuning MMS Adapter Models for Low-Resource, Multi-Lingual ASR


This repository provides scripts and utilities for fine-tuning Meta's Massive Multilingual Speech (MMS) adapter models for Automatic Speech Recognition (ASR), specifically for low-resource languages like Mooré.  

The fine-tuning process is based on using adapter layers for efficient training.


## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Training Instructions](#training-instructions)
- [Evaluation](#evaluation)
- [Inference](#inference)

## Features
- Fine-tuning MMS adapter models for ASR on low-resource languages.
- Support for loading data from Hugging Face datasets or local files.
- Connectionist Temporal Classification (CTC) loss for sequence-to-sequence ASR.
- Evaluation with word error rate (WER) metrics.

## Requirements
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Repository Structure
- `data/`: Data preprocessing scripts and local dataset handling.
- `models/`: Model loading, saving, and checkpoints.
- `scripts/`: Scripts for training, evaluation, tokenization, and inference.
- `utils/`: Configuration and helper functions.
- `logs/`: Logs for training and evaluation results.

## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mms-adapter-asr-finetuning.git
   cd mms-adapter-asr-finetuning
   ```
   
2. Set up the environment:
   ```bash
   ./setup.sh
   ```

3. Preprocess the data:
   ```bash
   python scripts/preprocess.py --data_source <huggingface/local>
   ```

4. Train the model:
   ```bash
   python scripts/train.py --config utils/config.yaml
   ```

## Training Instructions
To start training, ensure your dataset is ready, and run the `train.py` script:
```bash
python scripts/train.py --config utils/config.yaml
```
You can adjust the training parameters in `config.yaml`.

## Evaluation
Evaluate the fine-tuned model using:
```bash
python scripts/evaluate.py --checkpoint models/checkpoints/
```
This will compute the Word Error Rate (WER) and other metrics.

## Inference
To perform inference using the fine-tuned model:
```bash
python scripts/inference.py --audio_file <path-to-audio>
```


---
# Acknowledgements

* [MMS](https://huggingface.co/facebook/mms-tts) was proposed in [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516) by Vineel Pratap, Andros Tjandra, Bowen Shi and co. You can find more details about the supported languages and their ISO 639-3 codes in the [MMS Language Coverage Overview](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html),
and see all MMS-TTS checkpoints on the Hugging Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts).
* [Hugging Face 🤗 Transformers](https://huggingface.co/docs/transformers/index) for the model integration, [Hugging Face 🤗 Accelerate](https://huggingface.co/docs/accelerate/index) for the distributed code and [Hugging Face 🤗 datasets](https://huggingface.co/docs/datasets/index) for facilitating datasets access.
* [Fine-tuning MMS Adapter Models for Multi-Lingual ASR](https://huggingface.co/blog/mms_adapters)
