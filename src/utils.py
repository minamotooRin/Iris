import torch
import random
import json

def clear_json(text):
    # result = text.replace("\n", ", ")
    result = text.replace("json", "")
    result = result.replace("`", "")
    result = result.replace("{,  \"","{\"") 
    result = result.replace(", }", "}")
    result = result.replace(",,", ",")
    result = result.strip(", ")
    result = result.strip()
    return result

def string2bool(s):
    if type(s) == bool:
        return s
    
    if type(s) != str:
        return False

    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}

    s = s.lower()
    if s in true_values:
        return True
    
    return False

def save_to_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, default=int) + "\n")

def load_jsonl(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [json.loads(d) for d in data]

def seed_everything(seed = 42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

good_title = {
    'brazil': 'Brazil',
    'turkey': 'Turkey',
    'united-states': 'USA',
    'india': 'India',
    'japan': 'Japan',
    'portugal': 'Portugal',
    'nigeria': 'Nigeria'
}
countries = ['brazil', 'turkey', 'united-states', 'india', 'japan', 'portugal', 'nigeria']
