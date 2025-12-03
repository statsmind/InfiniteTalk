"""
模型管理器 - 预加载和管理 InfiniteTalk 模型实例
"""
import logging
import queue
import threading
from typing import Optional

import torch
import wan
from transformers import Wav2Vec2FeatureExtractor

from .config import GenerationConfig, ModelConfig
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from wan.configs import WAN_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ModelInstance:
    """单个模型实例"""

    def __init__(self, instance_id: int, device_id: int = 0):
        self.instance_id = instance_id
        self.device_id = device_id  # 设备编号（整数）
        self.pipeline = None
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None
        self.is_loaded = False

    def load(self):
        """加载模型"""
        try:
            logger.info(f"Loading model instance {self.instance_id} on cuda:{self.device_id}")

            # 获取配置
            cfg = WAN_CONFIGS[ModelConfig.TASK]

            # 初始化 Wav2Vec 音频编码器
            logger.info("Loading Wav2Vec2 audio encoder...")
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                ModelConfig.WAV2VEC_DIR, local_files_only=True
            ).to("cpu")  # 音频编码器放在 CPU 上
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                ModelConfig.WAV2VEC_DIR, local_files_only=True
            )

            # 初始化 InfiniteTalk pipeline
            logger.info("Loading InfiniteTalk pipeline...")
            self.pipeline = wan.InfiniteTalkPipeline(
                config=cfg,
                checkpoint_dir=ModelConfig.CKPT_DIR,
                quant_dir=ModelConfig.QUANT_DIR,
                device_id=self.device_id,  # 传入整数设备编号
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,
                lora_dir=[ModelConfig.LORA_DIR] if ModelConfig.LORA_DIR else None,
                lora_scales=GenerationConfig.LORA_SCALE,
                quant=GenerationConfig.QUANT,
                dit_path=ModelConfig.DIT_PATH,
                infinitetalk_dir=ModelConfig.INFINITETALK_DIR
            )

            # 启用显存管理
            if ModelConfig.NUM_PERSISTENT_PARAM_IN_DIT is not None:
                self.pipeline.vram_management = True
                self.pipeline.enable_vram_management(
                    num_persistent_param_in_dit=ModelConfig.NUM_PERSISTENT_PARAM_IN_DIT
                )

            self.is_loaded = True
            logger.info(f"Model instance {self.instance_id} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model instance {self.instance_id}: {e}")
            raise

    def unload(self):
        """卸载模型释放显存"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        if self.audio_encoder:
            del self.audio_encoder
            self.audio_encoder = None
        if self.wav2vec_feature_extractor:
            del self.wav2vec_feature_extractor
            self.wav2vec_feature_extractor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info(f"Model instance {self.instance_id} unloaded")


class ModelManager:
    """模型管理器 - 管理模型实例池"""

    def __init__(self, num_instances: int = 1, device_id: int = 0):
        self.num_instances = num_instances
        self.device_id = device_id  # GPU 设备编号
        self.instances = []
        self.available_queue = queue.Queue()
        self.lock = threading.Lock()
        self.is_initialized = False

    def initialize(self):
        """初始化所有模型实例"""
        if self.is_initialized:
            logger.warning("Model manager already initialized")
            return

        logger.info(f"Initializing {self.num_instances} model instance(s)...")

        for i in range(self.num_instances):
            instance = ModelInstance(instance_id=i, device_id=self.device_id)
            instance.load()
            self.instances.append(instance)
            self.available_queue.put(instance)

        self.is_initialized = True
        logger.info("Model manager initialized successfully")

    def acquire_instance(self, timeout: float = None) -> Optional[ModelInstance]:
        """获取一个可用的模型实例"""
        if not self.is_initialized:
            raise RuntimeError("Model manager not initialized")

        try:
            instance = self.available_queue.get(timeout=timeout)
            logger.info(f"Acquired model instance {instance.instance_id}")
            return instance
        except queue.Empty:
            logger.warning("No available model instance")
            return None

    def release_instance(self, instance: ModelInstance):
        """释放模型实例回池中"""
        if instance and instance in self.instances:
            self.available_queue.put(instance)
            logger.info(f"Released model instance {instance.instance_id}")

    def shutdown(self):
        """关闭模型管理器，释放所有资源"""
        logger.info("Shutting down model manager...")
        with self.lock:
            for instance in self.instances:
                instance.unload()
            self.instances.clear()
            # 清空队列
            while not self.available_queue.empty():
                try:
                    self.available_queue.get_nowait()
                except queue.Empty:
                    break
            self.is_initialized = False
        logger.info("Model manager shut down successfully")

    def get_stats(self) -> dict:
        """获取模型管理器统计信息"""
        return {
            "total_instances": len(self.instances),
            "available_instances": self.available_queue.qsize(),
            "is_initialized": self.is_initialized,
        }


# 全局模型管理器实例
# 从环境变量或配置中提取设备编号
device_str = ModelConfig.DEVICE
if ":" in device_str:
    device_id = int(device_str.split(":")[-1])
else:
    device_id = 0

model_manager = ModelManager(
    num_instances=ModelConfig.NUM_MODEL_INSTANCES,
    device_id=device_id
)