export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P --resume-download --local-dir-use-symlinks False
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base --resume-download --local-dir-use-symlinks False
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base --resume-download --local-dir-use-symlinks False
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk --resume-download --local-dir-use-symlinks False

