import os
import json
import requests
import time
import csv
import pandas as pd
import argparse
import requests
from requests.exceptions import RequestException
import urllib3
import logging
from time import sleep
import time
import openai
from openai import AzureOpenAI


API_KEY = os.getenv("OPENAI_API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("OPENAI_AZURE_ENDPOINT")


if not API_KEY:
    raise ValueError("API_KEY is not set")
if not API_VERSION:
    raise ValueError("API_VERSION is not set")
if not AZURE_ENDPOINT:
    raise ValueError("AZURE_ENDPOINT is not set")

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

model = "gpt-4o"

if __name__ == "__main__":
    # take argparser input
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./vlm-based/outputs/gpt4o', help="Path to the output directory of the target model")
    parser.add_argument("--input_dir", type=str, default='./outputs', help="Path to the input file")
    parser.add_argument("--local_image_dir", type=str, default='./downloaded_images', help="Path to the local directory for downloaded images")
    args = parser.parse_args()


    countries = ["brazil", "india", "japan", "nigeria", "portugal", "turkey", "united-states"]
    models = ["cap-edit", "e2e-instruct", "cap-retrieve"]
    # write path, category, sub_category and model response to a csv file

    for country in countries:
        for model in models:
            output_dir = f"{args.output_dir}/{country}/{model}"
            # Create a local directory for downloaded images
            local_image_dir = f"{args.local_image_dir}/{country}/{model}"
            os.makedirs(local_image_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            output_filepath = f"{output_dir}/visual_sim.csv"
            with open(output_filepath, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["Src Path", "Tgt Path", "Reasoning", "Score"])

            input_file = f"{args.input_dir}/{model}/{country}/metadata.csv"
            df = pd.read_csv(input_file)

            original_src_paths = df['src_image_path'].tolist()
            original_tgt_paths = df['tgt_image_path'].tolist()



            # Function to process files in batches
            def process_in_batches(files, batch_size):
                for i in range(0, len(files), batch_size):
                    end_idx = i+batch_size if i+batch_size <= len(files) else len(files)
                    batch = files[i:end_idx]
                    original_src_paths_batch = original_src_paths[i:end_idx]
                    original_tgt_paths_batch = original_tgt_paths[i:end_idx]
                    
                    
                    for src_path, tgt_path, file in zip(original_src_paths_batch, original_tgt_paths_batch, batch):
                        try:
                            category = src_path.split('/')[-1].split('_')[0]
                            country = tgt_path.split('/')[-2]
                            country = country.capitalize()

                            messages = [
                                {"role": "assistant", "content": "You are an expert in evaluating the cultural relevance of images."},
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"You will be given two images. You have to assess how culturally relevant both images are with respect to the culture of {country}."
                                        },
                                        {
                                            "type": "text",
                                            "text": f"Please explain your reasoning step by step for both images, specifically considering cultural symbols, styles, traditions, or any features that align with the culture of the speaking population of {country}.\n"
                                        },
                                        {
                                            "type": "text",
                                            "text": "For each image, the final score should be a number between 1 to 5, where 1 and 5 mean the following:\n1 = Not culturally relevant,\n5 = Culturally relevant.\n"
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": src_path  # Replace this with the actual URL or file reference for the first image
                                            }
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": tgt_path  # Replace this with the actual URL or file reference for the second image
                                            }
                                        },
                                        {
                                            "type": "text",
                                            "text": "The output should be a JSON object ONLY with the following format:\n{\"first_reasoning\": ..., \"first_score\": number, \"second_reasoning\": ..., \"second_score\": number}\n"
                                        }
                                    ]
                                }
                            ]

                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages,
                                max_tokens=300,
                            )


                            response = response.choices[0].message.content
                            result = response.replace("\n", ", ")
                            result = result.replace("json", "")
                            result = result.replace("`", "")
                            result = result.replace("{, \"","{\"") 
                            result = result.replace(", }", "}")
                            result = result.replace(",,", ",")
                            result = result.strip(", ")
                            result = result.strip()
                            print(result)
                            result = json.loads(result)
                            first_reasoning = result['first_reasoning']
                            first_score = result['first_score']
                            second_reasoning = result['second_reasoning']
                            second_score = result['second_score']
                        except Exception as e:
                            result = str(e)
                            first_reasoning = ""
                            first_score = 0
                            second_reasoning = ""
                            second_score = 0
                        with open(output_filepath, mode='a') as csv_file:
                            writer = csv.writer(csv_file)
                            # writer.writerow([file, src_path, tgt_path, caption, llm_edit, retrieved_caption, result]) 
                            writer.writerow([src_path, tgt_path, first_reasoning, first_score, second_reasoning, second_score])
                        
                        if end_idx % 10 == 0:
                            print(f"Processed {end_idx} files")

            # Process the files in batches of 10
            process_in_batches(original_tgt_paths, 1)

