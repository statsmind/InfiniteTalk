import os
import json
import uuid
import shutil
import logging
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

import torch
import wan
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import str2bool
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="InfiniteTalk API", description="API for InfiniteTalk: Audio-driven Video Generation", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
wan_i2v = None
wav2vec_feature_extractor = None
audio_encoder = None

class GenerationRequest(BaseModel):
    prompt: str
    size: str = "infinitetalk-480"
    sample_steps: int = 40
    sample_shift: Optional[float] = None
    text_guide_scale: float = 5.0
    audio_guide_scale: float = 4.0
    seed: int = 42
    motion_frame: int = 9
    max_frame_num: int = 1000
    audio_type: str = "para"  # para or add
    color_correction_strength: float = 1.0
    use_teacache: bool = False,
    use_apg: bool = False

class GenerationResponse(BaseModel):
    video_path: str
    message: str

def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    if not (left_path=='None' or right_path=='None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path=='None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path=='None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def audio_prepare_single(audio_path, sample_rate=16000):
    human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    return human_speech_array

@app.on_event("startup")
async def startup_event():
    global wan_i2v, wav2vec_feature_extractor, audio_encoder
    
    # 初始化模型
    logger.info("Initializing InfiniteTalk pipeline...")
    
    # 模型路径配置 - 需要根据实际环境进行调整
    ckpt_dir = os.getenv("CKPT_DIR", "./weights/Wan2.1-I2V-14B-480P")
    wav2vec_dir = os.getenv("WAV2VEC_DIR", "./weights/chinese-wav2vec2-base")
    infinitetalk_dir = os.getenv("INFINITETALK_DIR", "./weights/InfiniteTalk/single/infinitetalk.safetensors")
    
    # 检查模型路径是否存在
    if not os.path.exists(ckpt_dir):
        logger.warning(f"Checkpoint directory {ckpt_dir} does not exist")
    if not os.path.exists(wav2vec_dir):
        logger.warning(f"Wav2Vec directory {wav2vec_dir} does not exist")
    if not os.path.exists(infinitetalk_dir):
        logger.warning(f"InfiniteTalk directory {infinitetalk_dir} does not exist")
    
    try:
        cfg = WAN_CONFIGS["infinitetalk-14B"]
        wav2vec_feature_extractor, audio_encoder = custom_init('cpu', wav2vec_dir)
        
        wan_i2v = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            lora_dir=None,
            lora_scales=[1.2],
            quant=None,
            dit_path=None,
            infinitetalk_dir=infinitetalk_dir
        )
        
        logger.info("InfiniteTalk pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize InfiniteTalk pipeline: {e}")
        # 即使初始化失败也继续运行，允许后续重试

@app.get("/")
async def root():
    return {"message": "InfiniteTalk API is running"}

@app.get("/health")
async def health_check():
    if wan_i2v is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "starting", "model_loaded": False}

@app.post("/generate", response_model=GenerationResponse)
async def generate_video(
    prompt: str = Form(...),
    image: str = Form(...),
    audio1: str = Form(...),
    audio2: Optional[str] = Form(None),
    size: str = Form("infinitetalk-480"),
    sample_steps: int = Form(40),
    text_guide_scale: float = Form(5.0),
    audio_guide_scale: float = Form(4.0),
    seed: int = Form(42),
    motion_frame: int = Form(9),
    max_frame_num: int = Form(1000),
    audio_type: str = Form("para"),
    sample_shift: Optional[float] = Form(None),
):
    try:
        # 创建临时工作目录
        job_id = str(uuid.uuid4())
        work_dir = f"./temp/{job_id}"
        os.makedirs(work_dir, exist_ok=True)
        audio_save_dir = os.path.join(work_dir, "audio")
        os.makedirs(audio_save_dir, exist_ok=True)
        
        # 保存上传的文件
        image_path = os.path.join("uploads", image) # os.path.join(work_dir, "input_image.png")
        audio1_path = os.path.join("uploads", audio1) # os.path.join(work_dir, "audio1.wav")
        
        # with open(image_path, "wb") as f:
        #     shutil.copyfileobj(image.file, f)
        #
        # with open(audio1_path, "wb") as f:
        #     shutil.copyfileobj(audio1.file, f)
            
        audio2_path = None
        if audio2:
            audio2_path = os.path.join("uploads", audio2) # os.path.join(work_dir, "audio2.wav")
            # with open(audio2_path, "wb") as f:
            #     shutil.copyfileobj(audio2.file, f)
        
        # 构建输入数据
        input_data = {
            "prompt": prompt,
            "cond_video": image_path
        }
        
        # 处理音频
        if audio2_path:
            # 多人模式
            new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(
                audio1_path, audio2_path, audio_type
            )
            
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
            
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            emb2_path = os.path.join(audio_save_dir, '2.pt')
            sum_audio = os.path.join(audio_save_dir, 'sum.wav')
            
            sf.write(sum_audio, sum_human_speechs, 16000)
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            
            input_data["cond_audio"] = {
                "person1": emb1_path,
                "person2": emb2_path
            }
            input_data["video_audio"] = sum_audio
            input_data["audio_type"] = audio_type
        else:
            # 单人模式
            human_speech = audio_prepare_single(audio1_path)
            audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
            
            emb_path = os.path.join(audio_save_dir, '1.pt')
            sum_audio = os.path.join(audio_save_dir, 'sum.wav')
            
            sf.write(sum_audio, human_speech, 16000)
            torch.save(audio_embedding, emb_path)
            
            input_data["cond_audio"] = {
                "person1": emb_path
            }
            input_data["video_audio"] = sum_audio
        
        # 设置采样参数
        if sample_shift is None:
            if size == 'infinitetalk-480':
                sample_shift = 7
            elif size == 'infinitetalk-720':
                sample_shift = 11
            else:
                sample_shift = 7
            
        # 生成视频
        logger.info("Generating video...")
        video = wan_i2v.generate_infinitetalk(
            input_data,
            size_buckget=size,
            motion_frame=motion_frame,
            frame_num=81,
            shift=sample_shift,
            sampling_steps=sample_steps,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            seed=seed,
            offload_model=True,
            max_frames_num=max_frame_num,
            color_correction_strength=1.0,
            extra_args={
                "use_teacache": False,
                "teacache_thresh": 0.2
            },
        )
        
        # 保存视频
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        save_file = f"outputs/{job_id}_{formatted_prompt}_{formatted_time}"
        os.makedirs("outputs", exist_ok=True)
        
        save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        final_video_path = f"{save_file}.mp4"
        
        # 清理临时文件
        shutil.rmtree(work_dir, ignore_errors=True)
        
        return GenerationResponse(
            video_path=final_video_path,
            message="Video generated successfully"
        )
        
    except Exception as e:
        raise e
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_path}")
async def get_video(video_path: str):
    file_path = f"outputs/{video_path}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Video not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)