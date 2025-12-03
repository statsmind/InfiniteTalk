#!/bin/bash

# 启动 API 服务
echo "Starting InfiniteTalk API server..."
python -m uvicorn api.api_main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
