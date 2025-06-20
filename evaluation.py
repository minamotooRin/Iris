import json
import logging
import requests
import base64, json, time
import os
import unicodedata

from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

from src.utils import clear_json

logging.basicConfig(level=logging.INFO)

def load_image(image_path: str) -> str:
    """
    读取一张图片，返回 base64 编码字符串。

    由于 Ubuntu 文件系统上文件名可能以 NFD（拆分）形式存储，
    本函数会依次尝试 NFC（预组合）和 NFD（拆分）两种形式。
    """
    # 先把路径包装成 Path 对象
    p = Path(image_path)
    # 两种规范化形式
    candidates = [
        # 原样
        str(p),
        # NFC：预组合（é）
        unicodedata.normalize('NFC', str(p)),
        # NFD：拆分（e + ́）
        unicodedata.normalize('NFD', str(p)),
    ]

    last_err = None
    for path_str in candidates:
        try:
            with open(path_str, "rb") as f:
                data = f.read()
            # 成功读取后立即编码并返回
            return base64.b64encode(data).decode("utf-8")
        except FileNotFoundError as e:
            last_err = e
            # 如果这个形式不存在，就继续下一个
            continue

    # 如果所有形式都无法打开，就抛出最后一个错误
    raise last_err

def collect_image_metadata(root_dir: str = "/home/youyuan/Transcreation/Iris/download_images/image-transcreation/outputs/part1/zhang", source_dir: str = "/home/youyuan/Transcreation/Iris/download_images/image-transcreation/part1"):
    """
    遍历 root_dir 目录，收集所有图片及其相关元信息。
    假设目录结构：
    root_dir/
      ├── country_A/
      │     ├── Step1X/
      │     └── Gemini/
      ├── country_B/
      │     ├── Step1X/
      │     └── Gemini/
      ...
    图片文件名格式：<source>_<category>_<object>.<ext>
    """
    root = Path(root_dir)
    images = []

    for country_folder in root.iterdir():
        if not country_folder.is_dir():
            continue
        target_country = country_folder.name
        for step_folder in ["Step1X", "Gemini"]:
            folder = country_folder / step_folder
            if not folder.exists():
                continue
            for img_path in folder.iterdir():
                if not img_path.is_file():
                    continue
                # 只处理常见图片格式
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
                    continue

                # 解析文件名：source_category_object.ext
                name_parts = img_path.stem.split("_")
                if len(name_parts) < 3:
                    # 如果不符合预期格式，可以跳过或自行处理
                    continue
                source_country = name_parts[0]
                category = name_parts[1]
                # 剩下部分作为 object 名称（以防 object 名含下划线）
                object_name = "_".join(name_parts[2:])

                images.append({
                    "file_path": str(img_path),
                    "source_path": f"{source_dir}/{source_country}/{category}_{object_name}{img_path.suffix.lower()}",
                    "target_country": target_country,
                    "step": step_folder,
                    "source_country": source_country,
                    "category": category,
                    "object_name": object_name,
                })

    return images

class TranscreationEvaluator:
    def __init__(self, config):
        self.model_name = config["model_name"]
        self.api_key = config["api_key"]
        self.client = OpenAI(
            api_key=self.api_key,
        )

    def download_batch_results(self, file_id: str, save_path: str):
        """
        通过 REST API 下载 batch 输出文件内容并保存到本地
        """
        url = f"https://api.openai.com/v1/files/{file_id}/content"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        resp = requests.get(url, headers=headers, stream=True)
        resp.raise_for_status()
    
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8_192):
                f.write(chunk)

    def get_bodies(self, img1_path, img2_path, target_country, category = None):
        
        category_patch = f" and its category '{category}'"
        try:
            base64_image_1 = load_image(img1_path)
            base64_image_2 = load_image(img2_path)
        except Exception as e:
            logging.error(f"Error loading images: {e}")
            return {}

        instructions = {
            "culture-relevance": f"You are an expert in evaluating the cultural relevance of images.",
            "semantic": f"You are an expert in judging the semantic equality between images.",
            "visual": "You are an expert in judging the visual similarity between images."
        }

        prompt = {   
            "culture-relevance":[
                {"type": "text", "text": f"You will be given two images. You have to assess how culturally relevant both images are with respect to the culture of {target_country}."},
                {"type": "text", "text": f"Please explain your reasoning step by step for both images, specifically considering cultural symbols, styles, traditions, or any features that align with the culture of the speaking population of {target_country}.\n"},
                {"type": "text", "text": f"For each image, the final score should be a number between 1 to 5, where 1 and 5 mean the following:\n"},
                {"type": "text", "text": f"1 = Not culturally relevant,\n"},
                {"type": "text", "text": f"5 = Culturally relevant.\n"},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image_1}"}},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image_2}"}},
                {"type": "text", "text": f"The output should be a JSON object ONLY with the following format:\n"},
                {"type": "text", "text": "{\"first_reasoning\": ..., \"first_score\": number, \"second_reasoning\": ..., \"second_score\": number}\n"},
            ],
            "semantic":[
                {"type": "text", "text": f"Given the input image{category_patch}, determine if the comparison image belongs to the same category as the input.\n"},
                {"type": "text", "text": f"Please explain your reasoning step by step, and provide a final score at the end.\n"},
                {"type": "text", "text": f"The score should be between 1 and 5, where:\n"},
                {"type": "text", "text": f"1 = Dissimilar Category,\n"},
                {"type": "text", "text": f"5 = Same Category.\n"},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image_1}"}},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image_2}"}},
                {"type": "text", "text": f"The output should be a JSON object ONLY with the following format:\n"},
                {"type": "text", "text": "{\"reasoning\": ..., \"score\": number}\n"},
            ],
            "visual":[
                {"type": "text", "text": f"Given the input image, determine if there are visual changes in the second image as compared to the first."},
                {"type": "text", "text": f"Please explain your reasoning step by step, and provide a final score at the end."},
                {"type": "text", "text": f"The score should be between 1 and 5, where:\n1 = No visual change,\n5 = High visual changes."},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image_1}"}},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image_2}"}},
                {"type": "text", "text": "The output should be a JSON object ONLY with the following STRICT format: {\"reasoning\": ..., \"score\": number}"},
            ],
        }

        bodies = {}
        for key in ["culture-relevance", "semantic", "visual"]:

            history = [
                {
                    "role": "assistant", 
                    "content": [
                        { "type": "text", "text": instructions[key]}
                    ]
                },
                {
                    "role": "user", 
                    "content": prompt[key]
                }
            ]
                
            bodies[key] = {
                "model": self.model_name,
                "temperature": 0.0,
                "messages": history
            }
        
        return bodies

    def save_task_file(self, images, batch_file="batch.jsonl", record_file="batch_record.json"):
        tasks = []
        record = {}
        for item in tqdm(images):
            img1_path = item["source_path"]
            img2_path = item["file_path"]
            target_country = item["target_country"]
            category = item["category"]

            bodies = self.get_bodies(img1_path, img2_path, target_country, category)
            for key, body in bodies.items():
                custom_id = f"{item['source_country']}_{item['target_country']}_{item['category']}_{item['object_name']}_{key}"
                tasks.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                })
                record[custom_id] = item
        
        with open(batch_file, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(record_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4))
            
        print(f"Batch 作业文件已创建，包含 {len(tasks)} 个任务。")

    def sumbit_tasks(self, batch_file="batch.jsonl"):

        upload_resp = self.client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        file_id = upload_resp.id

        batch_job = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_id = batch_job.id
        print(f"Batch 作业已创建，file ID: {file_id}, 作业 ID: {batch_id}")

        return batch_job

    def delete_batch(self, batch_id):
        """
        删除指定的 Batch 作业。
        """
        try:
            self.client.files.delete(batch_id)
            print(f"Batch 作业 {batch_id} 已删除。")
        except Exception as e:
            print(f"删除 Batch 作业 {batch_id} 时出错：{e}")

    def query_results(self, batch_id, results_file=None):

        while True:
            status = self.client.batches.retrieve(batch_id)
            if status.status in ("completed", "failed"):
                break
            print(f"当前状态：{status.status}，等待中…")
            time.sleep(10)

        if status.status == "failed":
            print("Batch 作业验证或执行失败，错误信息：", status.errors)
            return status

        if results_file:

            results_file_id = status.output_file_id if status.output_file_id else status.error_file_id
            self.download_batch_results(results_file_id, results_file)
            print(f"处理完成，结果保存在{results_file}")

        return status
    
    def analysis(self, results_path):

        return
        for k,v in prompt.items():
            self.agent.reset_msg_history()
            try:
                response, text = self.agent.get_response(v, images = images)
                text = clear_json(text)
                result = json.loads(text)
                answer[k] = result
            except Exception as e:
                logging.error(f"Error processing {k} evaluation: {e}")

        return answer

def main(
    config_path: str = "configs/conf_GPT.json"
):
    config = json.load(open(config_path, "r"))
    config = config["model"]

    evaluator = TranscreationEvaluator(config)

    all_images = collect_image_metadata()
    # split images per N
    N = 16
    images_split = [all_images[i:i + N] for i in range(0, len(all_images), N)]

    # for i, images in enumerate(images_split):
    #     print(f"Processing batch {i + 1}/{len(images_split)} with {len(images)} images...")
    #     # Save each batch to a separate file
    #     batch_file = f"output/batch_{i + 1}.jsonl"
    #     record_file = f"output/batch_record_{i + 1}.json"
    #     evaluator.save_task_file(images, batch_file=batch_file, record_file=record_file)

    # images_split = [1]

    jobs = []
    for i in tqdm(range(len(images_split))):
        batch_file = f"output/batch_{i + 1}.jsonl"
        job = evaluator.sumbit_tasks(batch_file=batch_file)
        jobs.append(job)

    with open("output/batch_jobs.json", "w", encoding="utf-8") as f:
        f.write(json.dumps([job.to_dict() for job in jobs], ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main(config_path="configs/conf_GPT.json")  # Adjust the config path as needed