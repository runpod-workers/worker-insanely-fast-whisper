<div align="center">

<h1>Insanely-Fast-Whisper | Worker</h1>

[![CI | Test Handler](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_handler.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_handler.yml)
&nbsp;
[![CD | Build-Test-Release](https://github.com/runpod-workers/worker-template/actions/workflows/build-test-release.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/build-test-release.yml)

🚀 | Runpod worker for Insanely-Fast-Whisper.

</div>

#### Build an Image:

`docker build -t <your_dockerhub_directory>/image_name:tag`

Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand.

## Test Inputs

The following inputs can be used for testing the model:

```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "batch_size": 24, (Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24))
    "chunk_length": 30,
    "task": "transcribe", (Task to perform: transcribe or translate to another language. (default: transcribe))
    "language": None, (Language of the input audio. (default: "None" (Whisper auto-detects the language)))
  }
}
```

## Acknowledgments

- This tool is powered by Hugging Face's ASR models, primarily Whisper by OpenAI.
- Optimizations are developed by [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper).
