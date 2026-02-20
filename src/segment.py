import torch
import numpy as np
import gc
from PIL import Image
from transformers import SamProcessor, SamModel
from .config import Config
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

class InteriorSegmenter:
    """Wrapper for SAM model to perform click-based segmentation and memory management."""

    def __init__(self):
        """Initialize device, config settings, and load the SAM model."""
        self.device = Config.DEVICE
        self.sam_model_id = Config.MODEL_SAM_TYPE
        self.torch_dtype = Config.DTYPE_TORCH

        self.hash_image = None
        self.embeddings = None
        self.inputs = None

        self.model: Optional[SamModel] = None
        self.processor: Optional[SamProcessor] = None

        self.load_model()

    def load_model(self):
        """Load the SAM model to device if not already loaded."""
        if self.model is not None and self.processor is not None:
            return
        self.model = SamModel.from_pretrained(self.sam_model_id, torch_dtype = self.torch_dtype).to(self.device)
        self.processor = SamProcessor.from_pretrained(self.sam_model_id)


    def unload_model(self):
        """Free VRAM by deleting the predictor and clearing CUDA cache."""
        if self.model is not None:
            del self.model
            self.model = None
            logging.info('[SAM] Model unloaded')
        if self.processor is not None:
            del self.processor
            self.processor = None
            logging.info('[SAM] Processor unloaded')
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
            self.hash_image = None
            logger.info('[SAM] Embedding deleted successfully')

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info('[SAM] CUDA cash cleaned')


    def transfer_to_cpu(self):
        if self.model is None:
            logger.warning("[SAM] Can't transfer model. Model was not found")
            return
        device = next(self.model.parameters()).device

        if device.type == 'cuda':
                self.model = self.model.to("cpu")
                logger.info('[SAM] SAM moved to CPU')

                if self.embeddings is not None and self.embeddings.device.type == 'cuda':
                    self.embeddings=self.embeddings.to('cpu')
                    logger.info('[SAM] Embeddings moved to CPU')

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.info('[SAM] Model already in cpu')

    def transfer_to_gpu(self):
        if self.model is None:
            logger.warning("[SAM] Can't transfer model. Model was not found")
            return
        device = next(self.model.parameters()).device

        if device.type == 'cpu':
            self.model = self.model.to("cuda")
            logger.info('[SAM] SAM moved to CUDA')

            if self.embeddings is not None and self.embeddings.device.type == 'cpu':
                self.embeddings=self.embeddings.to('cuda')
                logger.info('[SAM] Embeddings moved to CUDA')

        else:
            logger.info('[SAM] Model already in CUDA')



    def _embeddings_define(self, image: Image.Image):
        self.load_model()
        if image is None:
            logger.warning('Image not found')
            return

        image_n = np.array(image)
        current_hash = hash(image_n.tobytes())

        if self.hash_image == current_hash and self.embeddings is not None:
            logger.info('[SAM] You are using same image')
            return
        else:
            logger.info("[SAM] You are using this image in first time")
            self.inputs = self.processor(image, return_tensors='pt')

            # to floaf16 if it possible
            for key, value in self.inputs.items():
                if isinstance(value, torch.Tensor):
                    self.inputs[key] = value.to(self.device)
                    if key == 'pixel_values':
                        self.inputs[key] = self.inputs[key].to(self.torch_dtype)

            self.hash_image = current_hash
            with torch.no_grad():
                self.embeddings = self.model.get_image_embeddings(
                    pixel_values= self.inputs['pixel_values']
                )




    def get_mask(self, image_p: Image.Image, point_coord: list):
        """
        Generate segmentation mask for a specific click point.

        Args:
            image_p: Input PIL image.
            point_coord: List of [x, y] coordinates.

        Returns:
             mask or None if image missing.
        """
        self.load_model()

        if image_p is None:
            logger.warning('Image not found')
            return None
        self._embeddings_define(image=image_p)

        original_size = self.inputs['original_sizes'].cpu().numpy()[0]  # [height, width]
        reshaped_size = self.inputs['reshaped_input_sizes'].cpu().numpy()[0]  # [1024, 1024]

        orig_h, orig_w = original_size
        reshaped_h, reshaped_w = reshaped_size

        x_orig, y_orig = point_coord
        x_new = x_orig * (reshaped_w / orig_w)
        y_new = y_orig * (reshaped_h / orig_h)

        point_tensor = torch.tensor(
            [[[[x_new, y_new]]]],
            dtype=torch.float32,
            device=self.device
        ).to(self.torch_dtype)

        label_tensor = torch.tensor(
            [[[1]]],
            dtype=torch.long,
            device=self.device
        )

        with torch.no_grad():
            outputs = self.model(
                image_embeddings=self.embeddings,
                input_points=point_tensor,
                input_labels=label_tensor,
                multimask_output=True
            )

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            self.inputs["original_sizes"].cpu(),
            self.inputs["reshaped_input_sizes"].cpu()
        )[0]
        if len(masks) == 0:
            logger.warning('[SAM] No mask in list')
            return None
        batch_mask = masks[0]
        if batch_mask.dim() == 4:
            batch_mask = batch_mask[0]

        areas = [np.sum(m.numpy()) for m in batch_mask]
        best_mask_idx = np.argmax(areas)

        if best_mask_idx >= batch_mask.shape[0]:
            logger.error(f'[SAM] Invalid mask index: {best_mask_idx} >= {batch_mask.shape[0]}')
            return None
        final_mask = batch_mask[best_mask_idx].numpy()
        if final_mask.ndim > 2:
            final_mask = final_mask.squeeze()
        if final_mask.sum() == 0:
            logger.warning('[SAM] Generated mask is empty (all zeros)')
            return None

        return final_mask