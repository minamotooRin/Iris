#!/bin/sh
#SBATCH -J train --qos epsrc -t 4-00:00:00
#SBATCH --nodes 1 --cpus-per-gpu 36
#SBATCH --gpus 1 --constraint=a100_80
source /bask/projects/x/xngs6460-languages/viyer/miniconda3/etc/profile.d/conda.sh
conda activate vision
python inference_hf.py --prompt_type "sem-eq" --model 'meta-llama/Llama-3.2-90B-Vision-Instruct' --country ${country} --vlm_name ${vlm_name}
python inference_hf.py --prompt_type "cultural-relevance" --model 'meta-llama/Llama-3.2-90B-Vision-Instruct' --country ${country} --vlm_name ${vlm_name}
