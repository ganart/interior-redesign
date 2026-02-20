import os
from dotenv import load_dotenv
import torch
from pathlib import Path
load_dotenv()
custom_hf_home = os.getenv('HF_HOME')

if custom_hf_home:
    os.environ['HF_HOME'] = custom_hf_home

class Config:
    """
    Configuration class containing project paths, model parameters, and design settings.
    """

    # ------- Paths --------------
    # Assuming config.py is in src/, parent.parent gives the root project dir
    BASE_DIR = Path(__file__).resolve().parent.parent

    OUTPUT_DIR = BASE_DIR / 'data' / 'output'
    INPUT_DIR = BASE_DIR / 'data' / 'input'
    MODELS_DIR = BASE_DIR / 'models'
    ASSETS_DIR = BASE_DIR / 'assets'  # Added assets directory

    # Create directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --------- Generation Parameters ---------
    IMAGE_SIZE = 1024
    INFERENCE_STEPS = 12
    GUIDANCE_SCALE = 2
    CONTROL_GUIDANCE_START = 0
    CONTROL_GUIDANCE_END = 0.6
    INPAINT_STRENGTH = 0.95

    SEED = None


    # ----------- Model Preferences _______
    MODEL_CONTROL_CANNY_ID = 'lllyasviel/sd-controlnet-canny'
    MODEL_CONTROL_DEPTH_ID = 'lllyasviel/sd-controlnet-depth'
    MODEL_INPAINT_ID = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
    MODEL_LoRa = 'latent-consistency/lcm-lora-sdv1-5'
    AUTOENCODER_ID = 'stabilityai/sd-vae-ft-mse'
    MODEL_SAM_TYPE = "facebook/sam-vit-large"


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use bfloat16 for CUDA to save VRAM, float32 for CPU
    if DEVICE == 'cuda':
        DTYPE_TORCH = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        DTYPE_TORCH = torch.float32

    # ControlNet influence strength
    CONTROL_STRENGTHS = {
        'canny': 0.5,
        'depth': 0.7
    }

    # ----------- Design Assets _______
    # Using absolute paths prevents "File not found" errors
    PATH_TO_BANNER = ASSETS_DIR / 'banner.png'
    EXMP_1 = ASSETS_DIR / 'example1.jpg'
    EXMP_2 = ASSETS_DIR / 'example2.png'
    EXMP_3 = ASSETS_DIR / 'example3.png'