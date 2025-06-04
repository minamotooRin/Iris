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
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def download_image(url, local_path, retries=3, backoff_factor=0.3):
    """Download image from a URL to a local file with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(local_path, 'wb') as file:
                    file.write(response.content)
                logging.info(f"Downloaded image from '{url}' to '{local_path}'")
                return True
            else:
                logging.info(f"Failed to download image from '{url}' - Status Code: {response.status_code}")
                return False
        except urllib3.exceptions.MaxRetryError as e:
            logging.info(f"MaxRetryError: {e}. Retrying...")
        except RequestException as e:
            logging.info(f"RequestException occurred: {e}. Retrying...")

        # Backoff before retrying
        sleep(backoff_factor * (2 ** attempt))
    
    logging.info(f"Failed to download image from '{url}' after {retries} retries.")
    return False


if __name__ == "__main__":
    # take argparser input
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./vlm-based/outputs', help="Path to the output directory of the target model")
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

            output_filepath = f"{output_dir}/cultural_relevance.csv"
            with open(output_filepath, mode='w') as file:
                writer = csv.writer(file)
                # writer.writerow(["Local Path", "Src Path", "Tgt Path", "Caption", "LLM Edit", "Retrieved Caption", "Model Response"])
                writer.writerow(["Src Path", "Tgt Path", "First Reasoning", "First Score", "Second Reasoning", "Second Score"])

            input_file = f"{args.input_dir}/{country}/{model}.csv"
            df = pd.read_csv(input_file)

            src_paths = df['Src path'].tolist()
            tgt_paths = df['Tgt path'].tolist()
            
            # Download images and prepare file paths
            local_image_paths = []
            final_src_paths = []
            final_tgt_paths = []
            for src_path, tgt_path in zip(src_paths, tgt_paths):
                base_path = tgt_path.split('/')[-1].split('.')[0]
                # get image extension from base path and append to local path
                img_ext = tgt_path.split('/')[-1].split('.')[1]
                local_path = os.path.join(local_image_dir, f"{base_path}.{img_ext}")
                if download_image(tgt_path, local_path):
                    local_image_paths.append(local_path)
                    final_src_paths.append(src_path)
                    final_tgt_paths.append(tgt_path)
                else:
                    continue


        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 512,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,

            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

        # Function to process files in batches
        def process_in_batches(files, batch_size):
            for i in range(0, len(files), batch_size):
                end_idx = i+batch_size if i+batch_size <= len(files) else len(files)
                batch = files[i:end_idx]
                src_paths_batch = final_src_paths[i:end_idx]
                tgt_paths_batch = final_tgt_paths[i:end_idx]
                src_uploaded_files = [upload_to_gemini(path, mime_type="image/jpeg") for path in src_paths_batch]
                tgt_uploaded_files = [upload_to_gemini(path, mime_type="image/jpeg") for path in tgt_paths_batch]
                
                for src_uploaded_file, tgt_uploaded_file, src_path, tgt_path, file in zip(src_uploaded_files, tgt_uploaded_files, src_paths_batch, tgt_paths_batch, batch):
                    try:
                        category = src_path.split('/')[-1].split('_')[0]
                        country = tgt_path.split('/')[-2]
                        country = country.capitalize()
                        prompt = [
                            f"Given the input image and its category '{category}', determine if there are visual changes in the second image as compared to the first.",
                            f"Please explain your reasoning step by step, and provide a final score at the end.",
                            "The score should be between 1 and 5, where:\n1 = No visual change,\n5 = High visual changes.",
                            src_uploaded_file,
                            tgt_uploaded_file,
                            "The output should be a JSON object ONLY with the following STRICT format: {\"reasoning\": ..., \"score\": number}",
                        ]

                        messages = [
                            {"role": "model", "parts": "You are an expert in judging the visual similarity between images."},
                            {"role": "user", "parts": prompt}
                        ]


                        response = model.generate_content(messages)
                        result = response.text.replace("\n", ", ")
                        result = result.replace("json", "")
                        result = result.replace("`", "")
                        result = result.replace("{,  \"","{\"") 
                        result = result.replace(", }", "}")
                        result = result.replace(",,", ",")
                        result = result.strip(", ")
                        result = result.strip()
                        print(result)
                        result = json.loads(result)
                        reasoning = result['reasoning']
                        score = result['score']
                    except Exception as e:
                        result = str(e)
                        first_reasoning = ""
                        first_score = 0
                        second_reasoning = ""
                        second_score = 0
                    with open(output_filepath, mode='a') as csv_file:
                        writer = csv.writer(csv_file)
                        # writer.writerow([file, src_path, tgt_path, caption, llm_edit, retrieved_caption, result]) 
                        writer.writerow([src_path, tgt_path, reasoning, score])
                # Sleep to avoid hitting rate limits
                time.sleep(5)


        # Process the files in batches of 10
        process_in_batches(tgt_paths, batch_size=10)
