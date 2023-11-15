import os
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline
)


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_pipeline(model, tokenizer, feature_extractor, torch_dtype, device):
    # Initialize the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model_kwargs={"use_flash_attention_2": True},
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe
# Define the model ID and check for GPU availability


def get_model(model_id, device, torch_dtype):
    model = fetch_pretrained_model(
        WhisperForConditionalGeneration,
        model_id,
        torch_dtype=torch_dtype
    ).to(device)
    tokenizer = WhisperTokenizerFast.from_pretrained(model_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    get_pipeline(model, tokenizer, feature_extractor, torch_dtype, device)
    return model, tokenizer, feature_extractor


if __name__ == "__main__":
    if os.environ.get("HF_HOME") != "/cache/huggingface":
        print(f"HF_HOME is set to {os.environ.get('HF_HOME')}")
        raise ValueError("HF_HOME must be set to /cache/huggingface")

    get_model("openai/whisper-large-v3",
              "cuda:0" if torch.cuda.is_available() else "cpu", torch.float16)
