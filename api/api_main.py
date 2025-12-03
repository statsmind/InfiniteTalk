"""
FastAPI 主服务 - InfiniteTalk API
"""
import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from einops import rearrange
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import GenerationConfig, ServerConfig, TaskConfig
from .file_handler import file_handler
from .model_manager import model_manager
from .task_queue import TaskStatus, task_queue
import warnings

torch.set_float32_matmul_precision('high')
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================
class TaskCreateResponse(BaseModel):
    """创建任务响应"""

    task_id: str
    status: str
    message: str


class TaskInfoResponse(BaseModel):
    """任务信息响应"""

    task_id: str
    prompt: str
    input_type: str
    audio_type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None


class TaskListResponse(BaseModel):
    """任务列表响应"""

    total: int
    tasks: List[TaskInfoResponse]


class StatsResponse(BaseModel):
    """统计信息响应"""

    queue_stats: dict
    model_stats: dict
    storage_stats: dict


# ==================== 辅助函数 ====================
def loudness_norm(audio_array, sr=16000, lufs=-23):
    """音频响度归一化"""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def audio_prepare_single(audio_path, sample_rate=16000):
    """准备单个音频"""
    human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    return human_speech_array


def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    """准备多人音频"""
    if not (left_path == "None" or right_path == "None"):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path == "None":
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path == "None":
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type == "para":
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type == "add":
        new_human_speech1 = np.concatenate(
            [
                human_speech_array1[: human_speech_array1.shape[0]],
                np.zeros(human_speech_array2.shape[0]),
            ]
        )
        new_human_speech2 = np.concatenate(
            [
                np.zeros(human_speech_array1.shape[0]),
                human_speech_array2[: human_speech_array2.shape[0]],
            ]
        )

    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000):
    """提取音频embedding"""
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device="cpu")
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )

    if len(embeddings) == 0:
        logger.error("Failed to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def process_task(task_id: str):
    """
    处理视频生成任务（后台任务）
    """
    logger.info(f"Starting to process task {task_id}")
    instance = None

    try:
        # 获取任务信息
        task = task_queue.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        # 更新任务状态为处理中
        task_queue.update_task_status(task_id, TaskStatus.PROCESSING)

        # 获取模型实例
        logger.info(f"Acquiring model instance for task {task_id}")
        instance = model_manager.acquire_instance(timeout=300)  # 5分钟超时
        if not instance:
            raise Exception("No available model instance")

        # 准备输入数据
        task_upload_dir = file_handler.get_task_upload_dir(task_id)
        task_audio_dir = file_handler.get_task_audio_dir(task_id)

        # 构建输入数据
        input_data = {}
        input_data["prompt"] = task.prompt

        # 获取输入图像或视频
        media_file = task_upload_dir / f"media{task.extra_params.get('media_ext', '')}"
        if not media_file.exists():
            raise Exception(f"Media file not found: {media_file}")
        input_data["cond_video"] = str(media_file)

        # 处理音频
        cond_audio = {}
        audio_type = task.audio_type

        if audio_type == "single":
            # 单人音频
            audio1_file = task_upload_dir / f"audio1{task.extra_params.get('audio1_ext', '')}"
            if not audio1_file.exists():
                raise Exception(f"Audio file not found: {audio1_file}")

            human_speech = audio_prepare_single(str(audio1_file))
            audio_embedding = get_embedding(
                human_speech,
                instance.wav2vec_feature_extractor,
                instance.audio_encoder,
            )
            emb_path = task_audio_dir / "1.pt"
            sum_audio = task_audio_dir / "sum.wav"
            sf.write(str(sum_audio), human_speech, 16000)
            torch.save(audio_embedding, str(emb_path))
            cond_audio["person1"] = str(emb_path)
            input_data["video_audio"] = str(sum_audio)

        else:
            # 多人音频
            audio1_file = task_upload_dir / f"audio1{task.extra_params.get('audio1_ext', '')}"
            audio2_file = task_upload_dir / f"audio2{task.extra_params.get('audio2_ext', '')}"

            if not audio1_file.exists() or not audio2_file.exists():
                raise Exception("Audio files not found")

            audio_type_mode = "para" if audio_type == "multi_para" else "add"
            input_data["audio_type"] = audio_type_mode

            new_human_speech1, new_human_speech2, sum_human_speechs = (
                audio_prepare_multi(
                    str(audio1_file), str(audio2_file), audio_type_mode
                )
            )

            audio_embedding_1 = get_embedding(
                new_human_speech1,
                instance.wav2vec_feature_extractor,
                instance.audio_encoder,
            )
            audio_embedding_2 = get_embedding(
                new_human_speech2,
                instance.wav2vec_feature_extractor,
                instance.audio_encoder,
            )

            emb1_path = task_audio_dir / "1.pt"
            emb2_path = task_audio_dir / "2.pt"
            sum_audio = task_audio_dir / "sum.wav"
            sf.write(str(sum_audio), sum_human_speechs, 16000)
            torch.save(audio_embedding_1, str(emb1_path))
            torch.save(audio_embedding_2, str(emb2_path))
            cond_audio["person1"] = str(emb1_path)
            cond_audio["person2"] = str(emb2_path)
            input_data["video_audio"] = str(sum_audio)

        input_data["cond_audio"] = cond_audio

        # 获取生成参数
        params = task.extra_params
        size = params.get("size", GenerationConfig.SIZE)
        sample_steps = params.get("sample_steps", GenerationConfig.SAMPLE_STEPS)
        sample_shift = params.get("sample_shift", GenerationConfig.SAMPLE_SHIFT)

        # 自动设置 sample_shift
        if sample_shift is None:
            if size == "infinitetalk-480":
                sample_shift = 7
            elif size == "infinitetalk-720":
                sample_shift = 11
            else:
                sample_shift = 7

        text_guide_scale = params.get(
            "text_guide_scale", GenerationConfig.SAMPLE_TEXT_GUIDE_SCALE
        )
        audio_guide_scale = params.get(
            "audio_guide_scale", GenerationConfig.SAMPLE_AUDIO_GUIDE_SCALE
        )
        mode = params.get("mode", GenerationConfig.MODE)
        motion_frame = params.get("motion_frame", GenerationConfig.MOTION_FRAME)
        frame_num = params.get("frame_num", GenerationConfig.FRAME_NUM)
        max_frame_num = params.get("max_frame_num", GenerationConfig.MAX_FRAME_NUM)
        seed = params.get("seed", GenerationConfig.BASE_SEED)
        use_teacache = params.get("use_teacache", GenerationConfig.USE_TEACACHE)
        teacache_thresh = params.get("teacache_thresh", GenerationConfig.TEACACHE_THRESH)
        use_apg = params.get("use_apg", GenerationConfig.USE_APG)
        color_correction_strength = params.get(
            "color_correction_strength", GenerationConfig.COLOR_CORRECTION_STRENGTH
        )

        # 构建 extra_args
        class Args:
            pass

        extra_args = Args()
        extra_args.use_teacache = use_teacache
        extra_args.teacache_thresh = teacache_thresh
        extra_args.use_apg = use_apg
        extra_args.apg_momentum = GenerationConfig.APG_MOMENTUM
        extra_args.apg_norm_threshold = GenerationConfig.APG_NORM_THRESHOLD
        extra_args.size = size

        # 生成视频
        logger.info(f"Generating video for task {task_id}")
        video = instance.pipeline.generate_infinitetalk(
            input_data,
            size_buckget=size,
            motion_frame=motion_frame,
            frame_num=frame_num,
            shift=sample_shift,
            sampling_steps=sample_steps,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            seed=seed,
            offload_model=True,
            max_frames_num=frame_num if mode == "clip" else max_frame_num,
            color_correction_strength=color_correction_strength,
            extra_args=extra_args,
        )

        # 保存结果
        from wan.utils.multitalk_utils import save_video_ffmpeg

        output_dir = file_handler.get_task_output_dir(task_id)
        output_file = output_dir / f"{task_id}"
        save_video_ffmpeg(
            video, str(output_file), [input_data["video_audio"]], high_quality_save=False
        )
        result_path = str(output_file) + ".mp4"

        # 更新任务状态为完成
        task_queue.update_task_status(
            task_id, TaskStatus.COMPLETED, result_path=result_path
        )
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        task_queue.update_task_status(
            task_id, TaskStatus.FAILED, error_message=str(e)
        )

    finally:
        # 释放模型实例
        if instance:
            model_manager.release_instance(instance)


# ==================== FastAPI 应用 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化模型
    logger.info("Initializing models...")
    model_manager.initialize()
    logger.info("Models initialized successfully")

    # 启动后台任务处理线程
    task_processor_thread = threading.Thread(
        target=background_task_processor, daemon=True
    )
    task_processor_thread.start()
    logger.info("Background task processor started")

    yield

    # 关闭时清理资源
    logger.info("Shutting down...")
    model_manager.shutdown()


app = FastAPI(
    title="InfiniteTalk API",
    description="Audio-driven Video Generation API for InfiniteTalk",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.ALLOW_ORIGINS,
    allow_credentials=ServerConfig.ALLOW_CREDENTIALS,
    allow_methods=ServerConfig.ALLOW_METHODS,
    allow_headers=ServerConfig.ALLOW_HEADERS,
)


def background_task_processor():
    """后台任务处理器"""
    logger.info("Background task processor running...")
    while True:
        try:
            # 获取下一个待处理任务
            task = task_queue.get_next_pending_task()
            if task:
                logger.info(f"Found pending task: {task.task_id}")
                process_task(task.task_id)
            else:
                # 没有待处理任务，休眠一会
                import time

                time.sleep(5)
        except Exception as e:
            logger.error(f"Error in background task processor: {e}")
            import time

            time.sleep(5)


# ==================== API 路由 ====================
@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "InfiniteTalk API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.post("/api/tasks", response_model=TaskCreateResponse)
async def create_task(
    prompt: str = Form(..., description="视频描述提示词"),
    media_file: str = File(..., description="输入的图像或视频文件"),
    audio1_file: str = File(..., description="第一个人的音频文件"),
    audio2_file: Optional[str] = File(None, description="第二个人的音频文件（多人时需要）"),
    audio_type: str = Form("single", description="音频类型: single, multi_add, multi_para"),
    size: str = Form(GenerationConfig.SIZE, description="分辨率: infinitetalk-480 或 infinitetalk-720"),
    sample_steps: int = Form(GenerationConfig.SAMPLE_STEPS, description="采样步数"),
    sample_shift: Optional[float] = Form(None, description="采样偏移"),
    text_guide_scale: float = Form(GenerationConfig.SAMPLE_TEXT_GUIDE_SCALE, description="文本引导比例"),
    audio_guide_scale: float = Form(GenerationConfig.SAMPLE_AUDIO_GUIDE_SCALE, description="音频引导比例"),
    mode: str = Form(GenerationConfig.MODE, description="生成模式: clip 或 streaming"),
    motion_frame: int = Form(GenerationConfig.MOTION_FRAME, description="运动帧数"),
    frame_num: int = Form(GenerationConfig.FRAME_NUM, description="帧数"),
    max_frame_num: int = Form(GenerationConfig.MAX_FRAME_NUM, description="最大帧数"),
    seed: int = Form(GenerationConfig.BASE_SEED, description="随机种子"),
    use_teacache: bool = Form(GenerationConfig.USE_TEACACHE, description="是否使用 TeaCache"),
    teacache_thresh: float = Form(GenerationConfig.TEACACHE_THRESH, description="TeaCache 阈值"),
    use_apg: bool = Form(GenerationConfig.USE_APG, description="是否使用 APG"),
    color_correction_strength: float = Form(GenerationConfig.COLOR_CORRECTION_STRENGTH, description="色彩校正强度"),
):
    """创建视频生成任务"""
    try:
        # 验证参数
        if audio_type not in ["single", "multi_add", "multi_para"]:
            raise HTTPException(status_code=400, detail="Invalid audio_type")

        if audio_type != "single" and not audio2_file:
            raise HTTPException(
                status_code=400, detail="audio2_file is required for multi-person mode"
            )

        # 创建任务
        media_ext = Path(media_file).suffix
        input_type = (
            "video"
            if media_ext.lower() in TaskConfig.ALLOWED_VIDEO_EXTENSIONS
            else "image"
        )

        task = task_queue.create_task(
            prompt=prompt,
            input_type=input_type,
            audio_type=audio_type,
            media_ext=media_ext,
            audio1_ext=Path(audio1_file).suffix,
            audio2_ext=Path(audio2_file).suffix if audio2_file else None,
            size=size,
            sample_steps=sample_steps,
            sample_shift=sample_shift,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            mode=mode,
            motion_frame=motion_frame,
            frame_num=frame_num,
            max_frame_num=max_frame_num,
            seed=seed,
            use_teacache=use_teacache,
            teacache_thresh=teacache_thresh,
            use_apg=use_apg,
            color_correction_strength=color_correction_strength,
        )

        # 保存上传的文件
        success, path, error = await file_handler.save_file(
            os.path.join(TaskConfig.UPLOAD_DIR, media_file), task.task_id, "media"
        )
        if not success:
            task_queue.delete_task(task.task_id)
            raise HTTPException(status_code=400, detail=error)

        success, path, error = await file_handler.save_file(
            os.path.join(TaskConfig.UPLOAD_DIR, audio1_file), task.task_id, "audio1"
        )
        if not success:
            file_handler.delete_task_files(task.task_id)
            task_queue.delete_task(task.task_id)
            raise HTTPException(status_code=400, detail=error)

        if audio2_file:
            success, path, error = await file_handler.save_file(
                os.path.join(TaskConfig.UPLOAD_DIR, audio2_file), task.task_id, "audio2"
            )
            if not success:
                file_handler.delete_task_files(task.task_id)
                task_queue.delete_task(task.task_id)
                raise HTTPException(status_code=400, detail=error)

        return TaskCreateResponse(
            task_id=task.task_id,
            status=task.status,
            message="Task created successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks", response_model=TaskListResponse)
async def get_tasks(
    status: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: int = 0,
):
    """获取任务列表"""
    try:
        status_filter = TaskStatus(status) if status else None
        tasks = task_queue.get_tasks(status=status_filter, limit=limit, offset=offset)

        task_responses = [
            TaskInfoResponse(
                task_id=task.task_id,
                prompt=task.prompt,
                input_type=task.input_type,
                audio_type=task.audio_type,
                status=task.status,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                error_message=task.error_message,
                result_path=task.result_path,
            )
            for task in tasks
        ]

        return TaskListResponse(
            total=len(task_queue.tasks),
            tasks=task_responses
        )
    except Exception as e:
        logger.error(f"Error getting tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}", response_model=TaskInfoResponse)
async def get_task(task_id: str):
    """获取任务信息"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskInfoResponse(
        task_id=task.task_id,
        prompt=task.prompt,
        input_type=task.input_type,
        audio_type=task.audio_type,
        status=task.status,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
        result_path=task.result_path,
    )


@app.get("/api/tasks/{task_id}/download")
async def download_task_result(task_id: str):
    """下载任务结果"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed, current status: {task.status}"
        )

    if not task.result_path or not Path(task.result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=task.result_path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4"
    )


@app.delete("/api/tasks/cleanup")
async def cleanup_tasks(days: int = TaskConfig.TASK_RETENTION_DAYS):
    """清理历史任务"""
    try:
        # 在后台线程执行清理操作，避免阻塞
        loop = asyncio.get_event_loop()

        # 清理任务队列
        deleted_tasks_count = await loop.run_in_executor(
            None, task_queue.cleanup_old_tasks, days
        )

        # 清理文件
        deleted_files_count, freed_space = await loop.run_in_executor(
            None, file_handler.cleanup_old_files, days
        )

        return {
            "message": "Cleanup completed successfully",
            "deleted_tasks": deleted_tasks_count,
            "deleted_files": deleted_files_count,
            "freed_space_mb": freed_space / 1024 / 1024,
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除任务及其相关文件"""
    logger.info(f"DELETE /api/tasks/{task_id} - Request received")

    task = task_queue.get_task(task_id)
    if not task:
        logger.warning(f"Task {task_id} not found")
        raise HTTPException(status_code=404, detail="Task not found")

    logger.info(f"Task {task_id} found, status: {task.status}")

    # 检查任务状态，禁止删除正在处理的任务
    if task.status == TaskStatus.PROCESSING:
        logger.warning(f"Attempted to delete task {task_id} while it is processing")
        raise HTTPException(
            status_code=400,
            detail="Cannot delete task that is currently being processed"
        )

    try:
        logger.info(f"Starting deletion process for task {task_id}")

        # 在后台线程执行删除操作，避免阻塞事件循环
        loop = asyncio.get_event_loop()

        # 删除文件
        logger.debug(f"Deleting files for task {task_id} in executor")
        file_delete_result = await loop.run_in_executor(
            None, file_handler.delete_task_files, task_id
        )
        logger.debug(f"File deletion result for task {task_id}: {file_delete_result}")

        # 删除任务记录
        logger.debug(f"Deleting task record for {task_id} from queue")
        queue_delete_result = task_queue.delete_task(task_id)
        logger.debug(f"Queue deletion result for task {task_id}: {queue_delete_result}")

        logger.info(f"Task {task_id} deleted successfully")
        return {"message": "Task deleted successfully", "task_id": task_id}

    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """获取系统统计信息"""
    return StatsResponse(
        queue_stats=task_queue.get_stats(),
        model_stats=model_manager.get_stats(),
        storage_stats=file_handler.get_storage_stats(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        workers=ServerConfig.WORKERS,
    )