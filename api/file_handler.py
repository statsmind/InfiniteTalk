"""
文件处理模块 - 处理文件上传、存储和管理
"""
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image

from .config import TaskConfig

logger = logging.getLogger(__name__)


class FileHandler:
    """文件处理器"""

    def __init__(self):
        self.upload_dir = Path(TaskConfig.UPLOAD_DIR)
        self.output_dir = Path(TaskConfig.OUTPUT_DIR)
        self.audio_save_dir = Path(TaskConfig.AUDIO_SAVE_DIR)

        # 确保目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_save_dir.mkdir(parents=True, exist_ok=True)

    async def save_file(
        self,
        file: str,
        task_id: str,
        file_type: str = "media"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        保存上传的文件

        Args:
            file: FastAPI UploadFile 对象
            task_id: 任务ID
            file_type: 文件类型 (media, audio1, audio2)

        Returns:
            (成功标志, 文件路径, 错误消息)
        """
        try:
            # 验证文件扩展名
            file_ext = Path(file).suffix.lower()
            if file_type == "media":
                allowed_exts = (
                    TaskConfig.ALLOWED_IMAGE_EXTENSIONS
                    | TaskConfig.ALLOWED_VIDEO_EXTENSIONS
                )
            else:  # audio
                allowed_exts = TaskConfig.ALLOWED_AUDIO_EXTENSIONS

            if file_ext not in allowed_exts:
                return False, None, f"不支持的文件格式: {file_ext}"

            # 创建任务专属目录
            task_dir = self.upload_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            filename = f"{file_type}{file_ext}"
            file_path = task_dir / filename

            # 保存文件
            with open(file_path, "wb") as f:
                with open(file, 'rb') as file_fp:
                    content = file_fp.read()
                    # 验证文件大小
                    if len(content) > TaskConfig.MAX_FILE_SIZE:
                        return False, None, f"文件大小超过限制 ({TaskConfig.MAX_FILE_SIZE / 1024 / 1024}MB)"
                    f.write(content)

            # 如果是图像文件，验证是否可以打开
            if file_ext in TaskConfig.ALLOWED_IMAGE_EXTENSIONS:
                try:
                    img = Image.open(file_path)
                    img.verify()
                except Exception as e:
                    os.remove(file_path)
                    return False, None, f"无效的图像文件: {str(e)}"

            return True, str(file_path), None

        except Exception as e:
            return False, None, f"保存文件失败: {str(e)}"

    async def save_upload_file(
        self,
        file: UploadFile,
        task_id: str,
        file_type: str = "media"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        保存上传的文件

        Args:
            file: FastAPI UploadFile 对象
            task_id: 任务ID
            file_type: 文件类型 (media, audio1, audio2)

        Returns:
            (成功标志, 文件路径, 错误消息)
        """
        try:
            # 验证文件扩展名
            file_ext = Path(file.filename).suffix.lower()
            if file_type == "media":
                allowed_exts = (
                    TaskConfig.ALLOWED_IMAGE_EXTENSIONS
                    | TaskConfig.ALLOWED_VIDEO_EXTENSIONS
                )
            else:  # audio
                allowed_exts = TaskConfig.ALLOWED_AUDIO_EXTENSIONS

            if file_ext not in allowed_exts:
                return False, None, f"不支持的文件格式: {file_ext}"

            # 创建任务专属目录
            task_dir = self.upload_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            filename = f"{file_type}{file_ext}"
            file_path = task_dir / filename

            # 保存文件
            with open(file_path, "wb") as f:
                content = await file.read()
                # 验证文件大小
                if len(content) > TaskConfig.MAX_FILE_SIZE:
                    return False, None, f"文件大小超过限制 ({TaskConfig.MAX_FILE_SIZE / 1024 / 1024}MB)"
                f.write(content)

            # 如果是图像文件，验证是否可以打开
            if file_ext in TaskConfig.ALLOWED_IMAGE_EXTENSIONS:
                try:
                    img = Image.open(file_path)
                    img.verify()
                except Exception as e:
                    os.remove(file_path)
                    return False, None, f"无效的图像文件: {str(e)}"

            return True, str(file_path), None

        except Exception as e:
            return False, None, f"保存文件失败: {str(e)}"

    def get_task_upload_dir(self, task_id: str) -> Path:
        """获取任务的上传目录"""
        return self.upload_dir / task_id

    def get_task_output_dir(self, task_id: str) -> Path:
        """获取任务的输出目录"""
        task_output_dir = self.output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        return task_output_dir

    def get_task_audio_dir(self, task_id: str) -> Path:
        """获取任务的音频embedding目录"""
        task_audio_dir = self.audio_save_dir / task_id
        task_audio_dir.mkdir(parents=True, exist_ok=True)
        return task_audio_dir

    def delete_task_files(self, task_id: str) -> bool:
        """删除任务相关的所有文件"""
        logger.info(f"Starting to delete files for task {task_id}")
        success = True

        try:
            # 删除上传文件
            upload_dir = self.upload_dir / task_id
            if upload_dir.exists():
                logger.debug(f"Deleting upload directory: {upload_dir}")
                shutil.rmtree(upload_dir, ignore_errors=False)
                logger.debug(f"Upload directory deleted: {upload_dir}")
            else:
                logger.debug(f"Upload directory does not exist: {upload_dir}")

            # 删除输出文件
            output_dir = self.output_dir / task_id
            if output_dir.exists():
                logger.debug(f"Deleting output directory: {output_dir}")
                shutil.rmtree(output_dir, ignore_errors=False)
                logger.debug(f"Output directory deleted: {output_dir}")
            else:
                logger.debug(f"Output directory does not exist: {output_dir}")

            # 删除音频embedding文件
            audio_dir = self.audio_save_dir / task_id
            if audio_dir.exists():
                logger.debug(f"Deleting audio directory: {audio_dir}")
                shutil.rmtree(audio_dir, ignore_errors=False)
                logger.debug(f"Audio directory deleted: {audio_dir}")
            else:
                logger.debug(f"Audio directory does not exist: {audio_dir}")

            logger.info(f"Successfully deleted all files for task {task_id}")
            return True

        except PermissionError as e:
            logger.error(f"Permission error deleting task {task_id} files: {e}")
            logger.error("Files may be in use by another process")
            return False
        except Exception as e:
            logger.error(f"Error deleting task {task_id} files: {e}", exc_info=True)
            return False

    def cleanup_old_files(self, days: int = None) -> Tuple[int, int]:
        """
        清理过期的文件

        Args:
            days: 保留天数

        Returns:
            (删除的任务数, 释放的空间字节数)
        """
        days = days or TaskConfig.TASK_RETENTION_DAYS
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_tasks = 0
        freed_space = 0

        # 清理上传文件
        for task_dir in self.upload_dir.iterdir():
            if task_dir.is_dir():
                # 检查目录修改时间
                dir_mtime = datetime.fromtimestamp(task_dir.stat().st_mtime)
                if dir_mtime < cutoff_date:
                    size = self._get_dir_size(task_dir)
                    shutil.rmtree(task_dir)
                    deleted_tasks += 1
                    freed_space += size

        # 清理输出文件
        for task_dir in self.output_dir.iterdir():
            if task_dir.is_dir():
                dir_mtime = datetime.fromtimestamp(task_dir.stat().st_mtime)
                if dir_mtime < cutoff_date:
                    size = self._get_dir_size(task_dir)
                    shutil.rmtree(task_dir)
                    freed_space += size

        # 清理音频embedding文件
        for task_dir in self.audio_save_dir.iterdir():
            if task_dir.is_dir():
                dir_mtime = datetime.fromtimestamp(task_dir.stat().st_mtime)
                if dir_mtime < cutoff_date:
                    size = self._get_dir_size(task_dir)
                    shutil.rmtree(task_dir)
                    freed_space += size

        return deleted_tasks, freed_space

    def _get_dir_size(self, path: Path) -> int:
        """计算目录大小"""
        total_size = 0
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size

    def get_storage_stats(self) -> dict:
        """获取存储统计信息"""
        upload_size = self._get_dir_size(self.upload_dir)
        output_size = self._get_dir_size(self.output_dir)
        audio_size = self._get_dir_size(self.audio_save_dir)

        return {
            "upload_size_mb": upload_size / 1024 / 1024,
            "output_size_mb": output_size / 1024 / 1024,
            "audio_size_mb": audio_size / 1024 / 1024,
            "total_size_mb": (upload_size + output_size + audio_size) / 1024 / 1024,
            "upload_tasks": len(list(self.upload_dir.iterdir())),
            "output_tasks": len(list(self.output_dir.iterdir())),
        }


# 全局文件处理器实例
file_handler = FileHandler()