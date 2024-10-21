import torch
from evaluate import load
from transformers import (
    Wav2Vec2ForCTC,
    TrainingArguments, 
    Trainer
)


from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import os


def train_model():
    # Implement your training logic here
    print("Training the MMS Adapter model...")


    wer_metric = load("wer")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    model.init_adapter_layers()

    # freeze all weights, but the adapter layers
    model.freeze_base_model()

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True


    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        num_train_epochs=5,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=200,
        eval_steps=100,
        logging_steps=100,
        learning_rate=1e-3,
        warmup_steps=100,
        save_total_limit=2,
        push_to_hub=True,
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
    adapter_file = os.path.join(training_args.output_dir, adapter_file)

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

    ### upload the result of the training to the 🤗 Hub.
    trainer.push_to_hub()

if __name__ == "__main__":
    train_model()
