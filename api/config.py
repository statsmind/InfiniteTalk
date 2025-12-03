"""
配置文件 - InfiniteTalk API 服务
"""
import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent.parent

# 模型配置
class ModelConfig:
    """模型路径配置"""
    # 基础模型路径
    CKPT_DIR = os.getenv("CKPT_DIR", str(BASE_DIR / "weights" / "Wan2.1-I2V-14B-480P"))

    # Wav2Vec 音频编码器路径
    WAV2VEC_DIR = os.getenv("WAV2VEC_DIR", str(BASE_DIR / "weights" / "chinese-wav2vec2-base"))

    # InfiniteTalk 权重路径
    INFINITETALK_DIR = os.getenv("INFINITETALK_DIR", str(BASE_DIR / "weights" / "InfiniteTalk" / "single" / "infinitetalk.safetensors"))

    # LoRA 权重路径（可选）
    LORA_DIR = None

    # 量化模型路径（可选）
    QUANT_DIR = os.getenv("QUANT_DIR", None)

    # DiT 路径（可选）
    DIT_PATH = os.getenv("DIT_PATH", None)

    # 任务类型
    TASK = "infinitetalk-14B"

    # 模型实例数量（用于并发处理）
    NUM_MODEL_INSTANCES = int(os.getenv("NUM_MODEL_INSTANCES", "1"))

    # 设备配置
    DEVICE = os.getenv("DEVICE", "cuda:0")

    # 显存管理参数
    NUM_PERSISTENT_PARAM_IN_DIT = int(os.getenv("NUM_PERSISTENT_PARAM_IN_DIT", "0"))

# 服务器配置
class ServerConfig:
    """API 服务器配置"""
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    WORKERS = int(os.getenv("API_WORKERS", "1"))
    RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

    # CORS 配置
    ALLOW_ORIGINS = ["*"]  # 允许所有域
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]

# 任务配置
class TaskConfig:
    """任务处理配置"""
    # 任务队列存储路径
    QUEUE_DB_PATH = os.getenv("QUEUE_DB_PATH", str(BASE_DIR / "task_queue.json"))

    # 上传文件存储路径
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))

    # 结果文件存储路径
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs"))

    # 音频embedding存储路径
    AUDIO_SAVE_DIR = os.getenv("AUDIO_SAVE_DIR", str(BASE_DIR / "audio_embeddings"))

    # 允许的文件扩展名
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
    ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}

    # 最大文件大小（单位：字节）
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(500 * 1024 * 1024)))  # 500MB

    # 任务历史保留天数
    TASK_RETENTION_DAYS = int(os.getenv("TASK_RETENTION_DAYS", "7"))

# 默认生成参数
class GenerationConfig:
    """视频生成默认参数"""
    # 分辨率
    SIZE = "infinitetalk-480"  # infinitetalk-480 或 infinitetalk-720

    # 帧数
    FRAME_NUM = 81
    MAX_FRAME_NUM = 1000

    # 采样参数
    SAMPLE_STEPS = 8
    SAMPLE_SHIFT = 2  # 会根据 size 自动设置
    SAMPLE_TEXT_GUIDE_SCALE = 1.0
    SAMPLE_AUDIO_GUIDE_SCALE = 2.0

    # 生成模式
    MODE = "streaming"  # clip 或 streaming
    MOTION_FRAME = 9

    # LoRA 参数
    LORA_SCALE = [1.0]

    # 其他参数
    BASE_SEED = 42
    COLOR_CORRECTION_STRENGTH = 1.0

    # 加速选项
    USE_TEACACHE = False
    TEACACHE_THRESH = 0.2
    USE_APG = False
    APG_MOMENTUM = -0.75
    APG_NORM_THRESHOLD = 55

    # 场景分割
    SCENE_SEG = False

    # 量化
    QUANT = os.getenv("QUANT", None)  # 'int8' 或 'fp8'

# 确保必要的目录存在
def ensure_directories():
    """确保所有必要的目录存在"""
    dirs = [
        Path(TaskConfig.QUEUE_DB_PATH).parent,
        Path(TaskConfig.UPLOAD_DIR),
        Path(TaskConfig.OUTPUT_DIR),
        Path(TaskConfig.AUDIO_SAVE_DIR),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# 在导入时自动创建目录
ensure_directories()