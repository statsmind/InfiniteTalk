"""
任务队列管理系统 - 使用本地 JSON 文件存储
"""
import json
import logging
import threading
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from .config import TaskConfig

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    """任务对象"""

    def __init__(
        self,
        task_id: str,
        prompt: str,
        input_type: str,
        audio_type: str = "single",
        status: TaskStatus = TaskStatus.PENDING,
        created_at: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        error_message: Optional[str] = None,
        result_path: Optional[str] = None,
        **kwargs
    ):
        self.task_id = task_id
        self.prompt = prompt
        self.input_type = input_type  # 'image' or 'video'
        self.audio_type = audio_type  # 'single', 'multi_add', 'multi_para'
        self.status = status
        self.created_at = created_at or datetime.now().isoformat()
        self.started_at = started_at
        self.completed_at = completed_at
        self.error_message = error_message
        self.result_path = result_path
        self.extra_params = kwargs

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "input_type": self.input_type,
            "audio_type": self.audio_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "result_path": self.result_path,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """从字典创建任务"""
        extra_params = data.pop("extra_params", {})
        return cls(**data, **extra_params)


class TaskQueue:
    """任务队列管理器"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or TaskConfig.QUEUE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()  # 使用可重入锁，避免递归死锁
        self._load_or_init()

    def _load_or_init(self):
        """加载或初始化队列数据"""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.tasks = {
                        task_id: Task.from_dict(task_data)
                        for task_id, task_data in data.items()
                    }
            except Exception as e:
                print(f"Error loading queue: {e}, initializing new queue")
                self.tasks = {}
        else:
            self.tasks = {}
        self._save()

    def _save(self):
        """保存队列到文件"""
        with self.lock:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(
                    {task_id: task.to_dict() for task_id, task in self.tasks.items()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    def create_task(
        self,
        prompt: str,
        input_type: str,
        audio_type: str = "single",
        **kwargs
    ) -> Task:
        """创建新任务"""
        task_id = str(uuid4())
        task = Task(
            task_id=task_id,
            prompt=prompt,
            input_type=input_type,
            audio_type=audio_type,
            status=TaskStatus.PENDING,
            **kwargs
        )
        with self.lock:
            self.tasks[task_id] = task
        self._save()
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务信息"""
        return self.tasks.get(task_id)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error_message: Optional[str] = None,
        result_path: Optional[str] = None,
    ):
        """更新任务状态"""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            task.status = status
            now = datetime.now().isoformat()

            if status == TaskStatus.PROCESSING:
                task.started_at = now
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completed_at = now

            if error_message:
                task.error_message = error_message
            if result_path:
                task.result_path = result_path

        self._save()
        return True

    def get_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Task]:
        """获取任务列表"""
        tasks = list(self.tasks.values())

        # 按创建时间倒序排序
        tasks.sort(key=lambda x: x.created_at, reverse=True)

        # 过滤状态
        if status:
            tasks = [t for t in tasks if t.status == status]

        # 分页
        if limit:
            tasks = tasks[offset : offset + limit]
        elif offset:
            tasks = tasks[offset:]

        return tasks

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        logger.info(f"Attempting to delete task {task_id}")
        try:
            with self.lock:
                logger.debug(f"Acquired lock for deleting task {task_id}")
                if task_id in self.tasks:
                    del self.tasks[task_id]
                    logger.debug(f"Task {task_id} removed from memory, saving to disk...")
                    self._save()
                    logger.info(f"Task {task_id} deleted successfully")
                    return True
                else:
                    logger.warning(f"Task {task_id} not found in queue")
                    return False
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}", exc_info=True)
            return False

    def cleanup_old_tasks(self, days: int = None) -> int:
        """清理历史任务"""
        days = days or TaskConfig.TASK_RETENTION_DAYS
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        with self.lock:
            task_ids_to_delete = []
            for task_id, task in self.tasks.items():
                task_date = datetime.fromisoformat(task.created_at)
                if task_date < cutoff_date:
                    task_ids_to_delete.append(task_id)

            for task_id in task_ids_to_delete:
                del self.tasks[task_id]
                deleted_count += 1

        if deleted_count > 0:
            self._save()

        return deleted_count

    def get_next_pending_task(self) -> Optional[Task]:
        """获取下一个待处理任务"""
        with self.lock:
            pending_tasks = [
                t for t in self.tasks.values() if t.status == TaskStatus.PENDING
            ]
            if pending_tasks:
                # 按创建时间排序，返回最早的
                pending_tasks.sort(key=lambda x: x.created_at)
                return pending_tasks[0]
        return None

    def get_stats(self) -> Dict:
        """获取队列统计信息"""
        stats = {
            "total": len(self.tasks),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                stats["pending"] += 1
            elif task.status == TaskStatus.PROCESSING:
                stats["processing"] += 1
            elif task.status == TaskStatus.COMPLETED:
                stats["completed"] += 1
            elif task.status == TaskStatus.FAILED:
                stats["failed"] += 1
        return stats


# 全局队列实例
task_queue = TaskQueue()