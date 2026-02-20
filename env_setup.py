import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

hf_home = os.getenv('HF_HOME')

if hf_home:
    os.environ['HF_HOME'] = hf_home

else:
    default = Path.home() / '.cache' / 'huggingface'
    default.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(default)


if hf_home and Path(hf_home).exists():
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'