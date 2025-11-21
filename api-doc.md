# InfiniteTalk API Documentation

The InfiniteTalk API allows you to generate talking avatar videos from images and audio inputs. This document explains how to use the API endpoints with examples.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

Check if the API is running and the model is loaded.

**Endpoint:** `GET /health`

#### Curl Example:
```bash
curl -X GET http://localhost:8000/health
```

#### Python Example:
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

#### Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Generate Video

Generate a talking avatar video from an image and audio input.

**Endpoint:** `POST /generate`

#### Parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| prompt | string | Yes | Text prompt describing the scene |
| image | file | Yes | Input image file (PNG/JPG) |
| audio1 | file | Yes | First audio file (WAV) |
| audio2 | file | No | Second audio file for multi-person mode (WAV) |
| size | string | No | Video size: `infinitetalk-480` (default) or `infinitetalk-720` |
| sample_steps | integer | No | Number of sampling steps (default: 40) |
| text_guide_scale | float | No | Text guidance scale (default: 5.0) |
| audio_guide_scale | float | No | Audio guidance scale (default: 4.0) |
| seed | integer | No | Random seed for generation (default: 42) |
| motion_frame | integer | No | Motion frame count (default: 9) |
| max_frame_num | integer | No | Maximum number of frames (default: 1000) |
| audio_type | string | No | Audio combination type: `para` (parallel) or `add` (sequential) |

#### Curl Example (Single Person):
```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=Hello, how are you today?" \
  -F "image=@input_image.png" \
  -F "audio1=@speech1.wav" \
  -F "size=infinitetalk-480" \
  -F "sample_steps=40" \
  -F "text_guide_scale=5.0" \
  -F "audio_guide_scale=4.0" \
  -F "seed=42" \
  -F "motion_frame=9" \
  -F "max_frame_num=1000" \
  -F "audio_type=para"
```

#### Curl Example (Multi-Person):
```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=Two people having a conversation" \
  -F "image=@input_image.png" \
  -F "audio1=@speech1.wav" \
  -F "audio2=@speech2.wav" \
  -F "size=infinitetalk-480" \
  -F "sample_steps=40" \
  -F "text_guide_scale=5.0" \
  -F "audio_guide_scale=4.0" \
  -F "seed=42" \
  -F "motion_frame=9" \
  -F "max_frame_num=1000" \
  -F "audio_type=para"
```

#### Python Example (Single Person):
```python
import requests

url = "http://localhost:8000/generate"
files = {
    "image": open("input_image.png", "rb"),
    "audio1": open("speech1.wav", "rb")
}
data = {
    "prompt": "Hello, how are you today?",
    "size": "infinitetalk-480",
    "sample_steps": 40,
    "text_guide_scale": 5.0,
    "audio_guide_scale": 4.0,
    "seed": 42,
    "motion_frame": 9,
    "max_frame_num": 1000,
    "audio_type": "para"
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result)

# Download the generated video
video_url = f"http://localhost:8000/video/{result['video_path'].split('/')[-1]}"
video_response = requests.get(video_url)
with open("generated_video.mp4", "wb") as f:
    f.write(video_response.content)
```

#### Python Example (Multi-Person):
```python
import requests

url = "http://localhost:8000/generate"
files = {
    "image": open("input_image.png", "rb"),
    "audio1": open("speech1.wav", "rb"),
    "audio2": open("speech2.wav", "rb")
}
data = {
    "prompt": "Two people having a conversation",
    "size": "infinitetalk-480",
    "sample_steps": 40,
    "text_guide_scale": 5.0,
    "audio_guide_scale": 4.0,
    "seed": 42,
    "motion_frame": 9,
    "max_frame_num": 1000,
    "audio_type": "para"
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result)
```

#### Response:
```json
{
  "video_path": "results/uuid_prompt_timestamp.mp4",
  "message": "Video generated successfully"
}
```

### 3. Retrieve Generated Video

Download a generated video by its filename.

**Endpoint:** `GET /video/{video_path}`

#### Curl Example:
```bash
curl -X GET http://localhost:8000/video/results/uuid_prompt_timestamp.mp4 -o output_video.mp4
```

#### Python Example:
```python
import requests

video_path = "results/uuid_prompt_timestamp.mp4"
response = requests.get(f"http://localhost:8000/video/{video_path.split('/')[-1]}")

with open("output_video.mp4", "wb") as f:
    f.write(response.content)
```

## Environment Variables

The API uses the following environment variables for model paths:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| CKPT_DIR | ./weights/Wan2.1-I2V-14B-480P | Main checkpoint directory |
| WAV2VEC_DIR | ./weights/chinese-wav2vec2-base | Wav2Vec model directory |
| INFINITETALK_DIR | ./weights/InfiniteTalk/single/infinitetalk.safetensors | InfiniteTalk model directory |

## Error Responses

The API may return the following error codes:

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Requested resource not found
- `500 Internal Server Error`: Server-side error during processing

Example error response:
```json
{
  "detail": "Error message describing the issue"
}
```