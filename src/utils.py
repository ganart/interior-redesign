import cv2

from .config import Config
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def load_image(img_name: str) -> Image.Image:
    """
    Load an image from the defined INPUT_DIR.

    Args:
        img_name (str): The filename of the image (e.g., 'room.jpg').

    Returns:
        Image.Image: PIL Image object converted to RGB.

    Raises:
        FileNotFoundError: If the image does not exist at the path.
    """
    path = Config.INPUT_DIR / img_name

    if not path.exists():
        logger.error(f'Image file not found: {path}')
        raise FileNotFoundError(f'{img_name} was not found in {Config.INPUT_DIR}')

    logger.info(f"Loading image: {img_name}")
    return Image.open(path).convert('RGB')


def save_image(img: Image.Image, file_name: str):
    """
    Save a PIL image to the defined OUTPUT_DIR.

    Args:
        img (Image.Image): The PIL image to save.
        file_name (str): The name of the file to save as.
    """
    # Ensure the output directory exists
    if not Config.OUTPUT_DIR.exists():
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = Config.OUTPUT_DIR / file_name
    try:
        img.save(path)
        logger.info(f'[Saved] Image successfully saved to: {path}')
    except Exception as e:
        logger.error(f"Failed to save image {file_name}: {e}")


def prepare_canny(img: Image.Image, sigma_value: float = 0.33) -> Image.Image:
    """
    Apply Canny edge detection to an image for ControlNet input.

    Args:
        img (Image.Image): Input PIL image.

    Returns:
        Image.Image: 3-channel RGB image representing the edges.
    """
    # Convert PIL to NumPy array
    c_img = np.array(img)

    if len(c_img.shape) == 3:
        gray = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = c_img

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    median = np.median(gray)

    lower = int(max(0, ((1.0 - sigma_value) * median)))
    upper = int(min(255, ((1.0 + sigma_value) * median)))
    # Apply Canny edge detection
    canny = cv.Canny(gray, lower, upper)

    kernel = np.ones((2,2), np.int8)
    thick_line = cv2.dilate(canny, kernel=kernel, iterations=1)

    # Merge into 3 channels (required for ControlNet input)
    canny_3c = cv.merge([thick_line, thick_line, thick_line])

    return Image.fromarray(canny_3c)


def get_depth(model, img: Image.Image) -> Image.Image:
    """
    Generate a depth map from an image using a depth estimation model.

    Args:
        model: Loaded depth estimation pipeline (HuggingFace).
        img (Image.Image): Input PIL image.

    Returns:
        Image.Image: Depth map converted to RGB.
    """
    # Inference
    results = model(img)

    # Extract depth map
    depth_image = results["depth"]

    return depth_image.convert('RGB')


def mask_prepare(mask: Image.Image, dilate_kernel: int = 15, padding_kernel: int = 25,
                 feather_strength: float = 0.5) -> Image.Image:
    mask_gen = mask.convert('L')

    mask_gen = mask_gen.filter(ImageFilter.GaussianBlur(radius=2))

    mask_array = np.array(mask_gen)
    _, binary = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((dilate_kernel, dilate_kernel), np.int8)
    img_dilation = cv2.dilate(binary, kernel=kernel, iterations=1)

    pad_k = np.ones((padding_kernel, padding_kernel), np.int8)
    img_padded = cv2.dilate(img_dilation, kernel=pad_k, iterations=1)

    dist = cv2.distanceTransform(img_padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    dist_norm = dist / (dist.max() + 1e-8)
    soft_mask = np.clip(dist_norm / feather_strength, 0, 1)
    soft_mask = (soft_mask * 255).astype(np.uint8)

    return Image.fromarray(soft_mask)
