import cv2

from .config import Config
from .utils import prepare_canny, get_depth, mask_prepare
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, LCMScheduler, AutoencoderKL
import gc
from PIL import Image, ImageFilter
import torch
from transformers import pipeline
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class InteriorInpaint:
    """
    Handles image inpainting using Stable Diffusion with ControlNet guidance.

    Attributes:
        device (str): Computation device (CPU or CUDA).
        pipe (StableDiffusionControlNetInpaintPipeline): The loaded diffusion pipeline.
        depth_model (Pipeline): HuggingFace pipeline for depth estimation.
    """

    def __init__(self):
        """Initialize models, configuration, and the depth estimator."""
        self.device = Config.DEVICE
        self.model_id = Config.MODEL_INPAINT_ID
        self.torch_dtype = Config.DTYPE_TORCH

        self.pipe: Optional[StableDiffusionControlNetInpaintPipeline] = None
        self.depth_model: Optional[pipeline] = None
        self.current_cnet_name: str = None

        # Mapping control names to model IDs
        self.control_id = {
            'canny': Config.MODEL_CONTROL_CANNY_ID,
            'depth': Config.MODEL_CONTROL_DEPTH_ID
        }

        self.depth_model_load()

    def load_model(self, control_model: str):
        """
        Load the Stable Diffusion pipeline with the specified ControlNet.

        Args:
            control_model (str): Type of control ('canny' or 'depth').

        Raises:
            ValueError: If the control_model type is unknown.
        """

        if control_model not in self.control_id:
            logger.error(f'Failed to load model. Unknown type: {control_model}')
            raise ValueError(f'Unknown control type: {control_model}')

        # Avoid reloading if the correct model is already active
        if self.pipe is not None and self.current_cnet_name == control_model:
            return

        model_value = self.control_id[control_model]
        self.unload_model()

        vae = AutoencoderKL.from_pretrained(
            Config.AUTOENCODER_ID,
            torch_dtype=self.torch_dtype
        )

        logger.info(f"Loading ControlNet: {model_value}")
        cnet = ControlNetModel.from_pretrained(
            model_value,
            torch_dtype=self.torch_dtype
        )

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            controlnet=cnet,
            vae=vae,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config,
        )
        try:
            self.pipe.load_lora_weights(Config.MODEL_LoRa)
            # self.pipe.fuse_lora()
            logger.info(f"[InpaintModel] LoRa fused successfully")
        except ValueError as e:
            logger.warning(f"[InpaintModel] Failed to load LoRa    {e}")

        self.pipe.enable_model_cpu_offload()

        self.current_cnet_name = control_model

    def depth_model_load(self):

        self.depth_model = pipeline(
            'depth-estimation',
            model='Intel/dpt-hybrid-midas',
            device='cpu'
        )

    def unload_model(self):
        """Unload the diffusion pipeline and clear GPU cache."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            gc.collect()
            torch.cuda.empty_cache()

    def generate(self, mask, image, prompt, control_type='depth', seed=None):
        """
        Run the inpainting generation process.

        Args:
            mask (PIL.Image): Binary mask image.
            image (PIL.Image): Original input image.
            prompt (str): Text description of the desired output.
            control_type (str): 'depth' or 'canny'.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            PIL.Image: The generated image resized to original dimensions.
        """
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(self.device).manual_seed(seed)

        # 1. Resize logic (Ensure dimensions are divisible by 8)
        w_orig, h_orig = image.size
        target_size_scale = Config.IMAGE_SIZE
        scale = target_size_scale / max(w_orig, h_orig)

        new_w = int(w_orig * scale) - int((w_orig * scale) % 8)
        new_h = int(h_orig * scale) - int((h_orig * scale) % 8)

        image_gen = image.convert('RGB').resize((new_w, new_h), Image.LANCZOS)
        mask_gen = mask.convert('L').resize((new_w, new_h), Image.NEAREST)
        mask_gen = mask_prepare(mask_gen)

        try:
            if control_type == 'canny':
                logger.info("[Inpaint] Creating Canny image...")
                control_image = prepare_canny(image_gen)
                strength = Config.CONTROL_STRENGTHS[control_type]
            elif control_type == 'depth':
                logger.info("[Inpaint] Creating Depth image...")
                if self.device == 'cuda':
                    self.depth_model.model.to(self.device)
                    self.depth_model.device = torch.device(self.device)

                control_image = get_depth(self.depth_model, image_gen)
                strength = Config.CONTROL_STRENGTHS[control_type]
                if self.device == 'cuda':
                    self.depth_model.model.to('cpu')
                    self.depth_model.device = torch.device("cpu")
                    torch.cuda.empty_cache()

        except:
            logger.error(f"Generation failed. Unknown control type: {control_type}")
            raise ValueError(f'Unknown control type: {control_type}')

        control_image = control_image.resize((new_w, new_h), Image.LANCZOS)

        # 3. Load Model and Generate
        self.load_model(control_model=control_type)

        pos_prompt = f'{prompt}, interior design, 8k, photorealistic, high quality, highly detailed'
        neg_prompt = "cartoon, 3d, disfigured, bad art, deformed, extra limbs, blur, watermark, text"

        # Use autocast for mixed precision inference
        enabled = (self.device == 'cuda')
        with torch.autocast(self.device, dtype=self.torch_dtype, enabled=enabled):
            gen_image = self.pipe(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                image=image_gen,
                mask_image=mask_gen,
                control_image=control_image,
                controlnet_conditioning_scale=strength,
                control_guidance_start=Config.CONTROL_GUIDANCE_START,
                control_guidance_end=Config.CONTROL_GUIDANCE_END,
                height=new_h,
                width=new_w,
                num_inference_steps=Config.INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                strength=Config.INPAINT_STRENGTH,
                generator=generator
            ).images[0]
        print(f"📊 VRAM used: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        # Resize back to original dimensions
        gen_image = gen_image.resize((w_orig, h_orig), Image.LANCZOS)

        return gen_image
