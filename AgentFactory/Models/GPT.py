import base64, json, time
from openai import OpenAI

def load_image(image_path):
    # load image to mime_type="image/jpeg" and base64 encode

    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    return b64_image

from AgentFactory.Models.LLM import MLLM_remote

class GPT(MLLM_remote):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name, api_key, instr = "You are a helpful assistant."):
        super().__init__(model_name, api_key)

        self.roleNames = {
            "system": "assistant",
            "user": "user",
            "assistant": "assistant"
        }

        self.instr = instr

        self.PRICE = {
            "gpt-4o":
            {
                "input": 2.5/1000000,
                "output": 10/1000000,
            },
            "openai-o3":
            {
                "input": 10/1000000,
                "output": 40/1000000,
            },
            "gpt-4.1":
            {
                "input": 2/1000000,
                "output": 8/1000000,
            },
        }

        self.client = OpenAI(
            api_key=api_key,
        )

    def set_instruction(self, instr):
        
        self.instr = instr
    
    def total_cost(self):
        return self.input_tokens * self.PRICE[self.model_name]["input"] + self.output_tokens * self.PRICE[self.model_name]["output"]
    
    def get_response(self, msgs, images, max_length = 0, reply_prefix = ""):

        """
        msgs: [
            {
                'role': 'assistant', 
                'content': [
                    {
                    'type': 'text', 
                    'text': 'You are a helpful assistant.'
                    }
                ]
            }, 
            {
                'role': 'user', 
                'content': [
                    {
                        'type': 'text', 
                        'text': 'Are the two images same?'
                    }, 
                    {'type': 'image'}, 
                    {'type': 'image'}
                ]
            }
        ]
        """

        instr = "".join([it["text"] for it in msgs[0]["content"]]) if msgs[0]["role"] == self.roleNames["system"] else ""
        self.set_instruction(instr)

        img_cnt = 0

        history = []
        for msg in msgs:

            demon = []

            for content in msg["content"]:
                if content["type"] == "image":
                    demon.append({
                        "type": "input_image" if msg["role"] == self.roleNames["user"] else "output_image",
                        "image_url": f"data:image/jpeg;base64,{load_image(images[img_cnt])}"
                    })
                    img_cnt += 1
                elif content["type"] == "text":
                    demon.append({
                        "type": "input_text" if msg["role"] == self.roleNames["user"] else "output_text",
                        "text": content["text"]
                    })

            history.append({"role": self.roleNames[msg["role"]], "content": demon})
        
        response = self.client.responses.create(
            model = self.model_name,
            input = history
        )
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

        return response, response.output_text
            
    def get_response(self, batch_msgs, images, max_length = 0, batch_file = "GPT_batch_input.jsonl", results_file = "GPT_batch_results.jsonl"):

        """
        msgs: [
            [
                {
                    'role': 'assistant', 
                    'content': [
                        {
                        'type': 'text', 
                        'text': 'You are a helpful assistant.'
                        }
                    ]
                }, 
                {
                    'role': 'user', 
                    'content': [
                        {
                            'type': 'text', 
                            'text': 'Are the two images same?'
                        }, 
                        {'type': 'image'}, 
                        {'type': 'image'}
                    ]
                }
            ],
            [...]
        ]
        """

        img_cnt = 0
        tasks = []
        for idx, msgs in enumerate(batch_msgs):
            history = []
            for msg in msgs:

                demon = []

                for content in msg["content"]:
                    if content["type"] == "image":
                        demon.append({
                            "type": "input_image" if msg["role"] == self.roleNames["user"] else "output_image",
                            "image_url": f"data:image/jpeg;base64,{load_image(images[img_cnt])}"
                        })
                        img_cnt += 1
                    elif content["type"] == "text":
                        demon.append({
                            "type": "input_text" if msg["role"] == self.roleNames["user"] else "output_text",
                            "text": content["text"]
                        })

                history.append({"role": self.roleNames[msg["role"]], "content": demon})
                
            task = {
                "custom_id": f"task-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "temperature": 0.0,
                    "messages": history
                }
            }
            tasks.append(task)

        with open(batch_file, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        # 上传文件（purpose 必须为 "batch"）
        upload_resp = self.client.files.upload(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        file_id = upload_resp.id  # 用于创建 batch 作业 :contentReference[oaicite:0]{index=0}

        # 创建 Batch 作业
        batch_job = self.client.batches.create(
            model=self.model_name,
            input=file_id,
            # batch 默认 processing_window=24h，可不显式指定
        )
        batch_id = batch_job.id

        # 轮询直到完成
        while True:
            status = self.client.batches.get(id=batch_id)
            if status.status in ("completed", "failed"):
                break
            print(f"当前状态：{status.status}，等待中…")
            time.sleep(10)

        if status.status == "failed":
            print("Batch 作业验证或执行失败，错误信息：", status.errors)
            return status

        # 下载结果和错误文件
        result_file = status.result  # 此字段包含结果文件 ID
        # 保存到本地
        self.client.files.download(result_file, stream=True, path=results_file)
        print("处理完成，结果保存在 batch_results.jsonl")

        return status

