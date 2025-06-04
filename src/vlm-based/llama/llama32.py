import os
import json
import requests
import csv
import pandas as pd
import argparse
import torch
import glob

from tqdm import tqdm
from time import sleep
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, MllamaForConditionalGeneration
from PIL import Image
from requests.exceptions import RequestException


SEM_EQ_PROMPT = """
    Given the input (first) image and its category '{category}', determine if the comparison (second) image belongs to the same category as the input.\n
    Please explain your reasoning step by step, and provide a final score at the end.\n
    The score should be between 1 and 5, where:\n
    1 = Dissimilar Category,\n
    5 = Same Category.\n
    The output should be a JSON object ONLY with the following format:\n
    {{'reasoning': ..., 'score': score}}
    ```json"""

CULT_RELEVANCE_PROMPT = """You will be given two images. You have to assess how culturally relevant both images are with respect to the culture of {country}.
Please explain your reasoning step by step for both images, specifically considering cultural symbols, styles, traditions, or any features that align with the culture of the speaking population of {country}.\n
For each image, the final score should be a number between 1 to 5, where 1 and 5 mean the following:\n
1 = Not culturally relevant,\n
5 = Culturally relevant.\n
{}
The output should be a JSON object ONLY with the following format:\n
{{'first_reasoning': ..., 'first_score': number, 'second_reasoning': ..., 'second_score': number}}
"""


def postprocess(text):
    result = text.split("```")[0]
    result = replace("\n", ", ")
    result = result.replace("{,  \"","{\"") 
    result = result.replace(", }", "}")
    result = result.replace(",,", ",")
    result = result.strip(", ")
    result = result.strip()
    try:
        result = json.loads(generated_text)
        reasoning = result['reasoning']
        score = result['score']
    except Exception as e:
        result = str(e)
        reasoning = ""
        score = 0
    return reasoning, score

if __name__ == "__main__":
    # take argparser input
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./vlms_eval/outputs/llama32', help="Path to the output directory of the target model")
    parser.add_argument("--input_dir", type=str, default='./outputs', help="Path to the input file")
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__)).split("automatic-eval-transcreation")[0]
    args.output_dir = os.path.join(dir_path, args.output_dir)
    args.input_dir = os.path.join(dir_path, args.input_dir)

    countries = ['brazil', 'india', 'japan', 'nigeria', 'portugal', 'turkey', 'united-states']
    models = ["cap-edit", "e2e-instruct", "cap-retrieve"]
    # write path, category, sub_category and model response to a csv file
    
    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    for country in countries:
        for vlm_name in models:
            output_dir = f"{args.output_dir}/{country}/{vlm_name}"
            os.makedirs(output_dir, exist_ok=True)

            output_filepath = f"{output_dir}/sem_eq.csv"
            with open(output_filepath, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["Src Path", "Tgt Path", "Reasoning", "Score"])

            input_file = f"{args.input_dir}/{vlm_name}/{country}/metadata.csv"
            df = pd.read_csv(input_file)

            final_src_paths = df['src_image_path'].tolist()
            final_tgt_paths = df['tgt_image_path'].tolist()
            
            for src_path, tgt_path in tqdm(zip(final_src_paths, final_tgt_paths)):
                # Process the images and text
                image_urls = [src_path, tgt_path]
                images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
                
                category = src_path.split('/')[-1].split('_')[0]

                inputs = processor.process(
                    images=images,
                    text=prompt.format(category=category)
                )
                # Move inputs to the correct device and make a batch of size 1
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

                # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
                    output = model.generate_from_batch(
                        inputs,
                        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                        tokenizer=processor.tokenizer
                    )

                # only get generated tokens; decode them to text
                generated_tokens = output[0,inputs['input_ids'].size(1):]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                reasoning, score = postprocess(generated_text)
                with open(output_filepath, mode='a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([src_path, tgt_path, reasoning, score])
