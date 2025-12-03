# 创建任务
curl -X POST http://localhost:18001/api/tasks -F "prompt=教授在讲课" -F "media_file=ref_image.png" -F "audio1_file=1.wav"

# 查看所有任务执行进度
curl http://localhost:18001/api/tasks

# 查看任务执行进度
curl http://localhost:18001/api/tasks/06b072eb-d2a1-4ed7-9161-177771870f7c

