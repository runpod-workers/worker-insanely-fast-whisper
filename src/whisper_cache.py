import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline
)

# Define the model ID and check for GPU availability
model_id = "openai/whisper-large-v3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

# Load the model, tokenizer, and feature extractor
model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype
).to(device)
tokenizer = WhisperTokenizerFast.from_pretrained(model_id)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

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

# Parameters for prediction
audio_path = "sam_altman_lex_podcast_367.flac"
task = "transcribe"  # or "translate" for translation
language = None  # specify the language or None for auto-detection
batch_size = 24
return_timestamps = True

# Run the prediction
outputs = pipe(
    audio_path,
    chunk_length_s=30,
    batch_size=batch_size,
    generate_kwargs={"task": task, "language": language},
    return_timestamps=return_timestamps,
)

print(outputs)
