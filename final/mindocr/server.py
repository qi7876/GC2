import os
import shutil
import subprocess
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 配置参数
PREDICT_SCRIPT = "/home/qi7876/projects/GC2/final/mindocr/tools/infer/text/predict_system.py"  # 确保路径正确
DETECTION_ALGORITHM = "DB++"
RECOGNITION_ALGORITHM = "CRNN_CH"
CHAR_DICT = "/home/qi7876/projects/GC2/final/mindocr/mindocr/utils/dict/ch_dict.txt"
SYSTEM_RESULTS_PATH = "/home/qi7876/projects/GC2/final/mindocr/inference_results/system_results.txt"

class ProcessedText(BaseModel):
    processed_text: str

@app.post("/upload-images/", response_model=ProcessedText)
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="未上传任何文件。")
    
    # 创建一个临时目录用于存放上传的图片
    temp_dir = "/home/qi7876/projects/GC2/final/mindocr/inference_images"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        # 保存所有上传的图像到临时目录
        for file in files:
            saved_filename = file.filename
            saved_path = os.path.join(temp_dir, saved_filename)
            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 调用推理脚本
        command = [
            "python",
            PREDICT_SCRIPT,
            "--image_dir",
            temp_dir,
            "--det_algorithm",
            DETECTION_ALGORITHM,
            "--rec_algorithm",
            RECOGNITION_ALGORITHM,
            "--rec_char_dict_path",
            CHAR_DICT,
        ]

        # 运行推理脚本
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")

        if not os.path.exists(SYSTEM_RESULTS_PATH):
            raise HTTPException(status_code=500, detail="system_results.txt not found.")

        # 读取 system_results.txt
        with open(SYSTEM_RESULTS_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()

        processed_text = process_text(raw_text)

        return ProcessedText(processed_text=processed_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        shutil.rmtree(temp_dir)

def process_text(raw_text: str) -> str:
    try:
        lines = raw_text.strip().split("\n")
        processed_lines = []

        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            image_filename, json_str = parts

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            all_texts = []

            for item in data:
                transcription = item.get("transcription", "")
                points = item.get("points", [])
                if not transcription or not points or len(points) < 1:
                    continue

                first_point = points[0]
                if len(first_point) < 2:
                    continue

                x, y = first_point[:2]
                all_texts.append({
                    "transcription": transcription,
                    "x": x,
                    "y": y
                })

            if not all_texts:
                processed_lines.append("")
                continue

            sorted_texts = sorted(all_texts, key=lambda item: (item["y"], item["x"]))

            processed_text = " ".join([item["transcription"] for item in sorted_texts])

            processed_lines.append(processed_text)

        return "\n".join(processed_lines)

    except Exception as e:
        return ""