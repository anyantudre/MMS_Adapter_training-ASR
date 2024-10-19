from datasets import Audio


def perform_inference(
        sample_audio,
        target_lang,
):
    print("Performing inference with the fine-tuned model...")
    model_id = "anyantudre/wav2vec2-large-mms-1b-mos-V2"

    model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang="mos").to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(model_id)

    processor.tokenizer.set_target_lang("mos")


    input_dict = processor(sample_test[rand_int]["audio"]["array"], sampling_rate=16_000, return_tensors="pt", padding=True)

    logits = model(input_dict.input_values.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]


    return processor.decode(pred_ids)

if __name__ == "__main__":
    perform_inference()
