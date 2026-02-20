from .inpainter import InteriorInpaint
from .config import Config
from .segment import InteriorSegmenter
import gradio as gr
import base64
import os
from PIL import Image
import numpy as np
import logging


logger = logging.getLogger(__name__)
class WebUi:
    """
        Main class for the Gradio Web Interface.

        Handles the UI layout, user interactions, and coordination between
        the segmentation model (SAM) and the inpainting model.
        """
    def __init__(self):
        try:
            self.inpaint = InteriorInpaint()
        except Exception as e:
            raise ValueError(f'[UI] Inpaint model was not found: {e}')
        try:
            self.segmenter = InteriorSegmenter()

        except Exception as e:
            raise ValueError(f'[UI] SAM model was not found: {e}')

    def _generate_gradient(self, width, height):
        """Creates a gradient texture for the mask overlay."""
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)
        gradient_map = (xv + yv) / 2
        color1 = np.array([255, 0, 255, 160])
        color2 = np.array([0, 255, 255, 160])
        grad = (1 - gradient_map)[..., None] * color1 + gradient_map[..., None] * color2
        return grad.astype(np.uint8)

    def _apply_mask_overlay(self, image_pil, mask_np):
        """
                Draws a visual mask overlay on top of the image for the UI.

                Args:
                    image_pil (PIL.Image): The base image.
                    mask_np (np.array): The binary mask to overlay.

                Returns:
                    PIL.Image: Image with the colorful mask overlay.
                """
        if mask_np is None: return image_pil
        image_rgba = image_pil.convert("RGBA")
        w, h = image_rgba.size

        # Gradient
        gradient = self._generate_gradient(w, h)
        overlay_np = np.zeros((h, w, 4), dtype=np.uint8)

        # Sizing (just in case)
        h_mask, w_mask = mask_np.shape[:2]
        if h_mask != h or w_mask != w:
            mask_pil = Image.fromarray(mask_np.astype(np.uint8))
            mask_np = np.array(mask_pil.resize((w, h), Image.NEAREST))

        overlay_np[mask_np > 0] = gradient[mask_np > 0]
        return Image.alpha_composite(image_rgba, Image.fromarray(overlay_np)).convert("RGB")

    def predict(self, image, prompt, control_mode, current_mask):
        """
                Main generation function called by the 'Generate' button.

                Args:
                    image (PIL.Image): The input image.
                    prompt (str): User's design description.
                    control_mode (str): ControlNet mode ('depth' or 'canny').
                    current_mask (np.array): The segmentation mask from State.

                Returns:
                    tuple: (Generated Image, Gradio Group update to show results).
                """
        if self.inpaint is None or image is None: return None
        if current_mask is None:
            logger.warning("[UI] Error: No mask selected")
            return image
        self.segmenter.transfer_to_cpu()
        gr.Info("🎨 Generating... This may take 20-30 seconds")
        # Mask from State (0/1) -> PIL (0/255)
        mask_pil = Image.fromarray((current_mask * 255).astype('uint8')).convert('L')

        mask_res = mask_pil.resize((image.size[0], image.size[1]), Image.NEAREST)
        gen_image = self.inpaint.generate(
            mask=mask_res,
            image=image,
            prompt=prompt,
            control_type=control_mode,
            seed=Config.SEED
        )
        return gen_image, gr.Group(visible=True)

    def on_upload(self, image):
        """Callback when an image is uploaded."""
        if image is not None:
            self.segmenter._embeddings_define(image)
            logger.info('[SAM] Embeddings was created successfully')
        return image, None

    def on_click(self, original_image, mask_state, evt: gr.SelectData):
        """
                Callback when the user clicks on the image.
                Runs SAM to segment the object at the clicked coordinates.
        """
        if original_image is None: return None, mask_state
        self.segmenter.transfer_to_gpu()
        x, y = evt.index

        new_mask = self.segmenter.get_mask(original_image, point_coord=[x, y])
        if new_mask is None: return original_image, mask_state

        new_mask_np = np.array(new_mask)

        # Combining masks
        if mask_state is None:
            updated_mask = new_mask_np
        else:
            if mask_state.shape == new_mask_np.shape:
                updated_mask = np.maximum(mask_state, new_mask_np)
            else:
                updated_mask = new_mask_np

        visual_result = self._apply_mask_overlay(original_image, updated_mask)
        return visual_result, updated_mask


    def clear_mask(self, original_image, mask_state):
        """Callback to clear the current selection mask."""
        if original_image is None: return None, mask_state

        if mask_state is None: return original_image, mask_state
        new_mask_np = np.zeros_like(mask_state)
        return original_image, new_mask_np


    def encode_to_base64(self, file_path):
        """
                Reads an image file and converts it to a Base64 string for HTML embedding.

                Args:
                    file_path (str): Path to the image file.

                Returns:
                    str: Base64 data string or empty string on error.
        """
        if not os.path.exists(file_path):
            logger.error(f"[UI Warning] Banner was not found in this way: {file_path}")
            return ""  # return the emptiness so that it doesn't crumble.
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

    def create_demo(self):
        """Builds and configures the Gradio Blocks interface."""
        # --- CSS ---
        self.css = """

        body, .gradio-app { padding: 0 !important; margin: 0 !important; }

        .gradio-container {
            max-width: 100% ;
            padding: 0 ;
            margin: 0 ;
            gap: 0 ;
            overflow-x: hidden;
            background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
        }

        /* Remove all indents */
        #banner-component,
        #banner-component > div,
        #banner-component .prose {
            padding: 0 ;
            margin: 0 ;
            border: none ;
            background: none ;
            max-width: 100% ;
        }

        /* Banner container */
        .banner-container {
            width: 100vw;
            height: 550px;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
            margin-top: -1.1vw;
            overflow: hidden;
            z-index: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Picture as background */
        .banner-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: brightness(0.75) contrast(1.1);
            z-index: 1;
        }

        /* Gradient overlay */
        .banner-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.3) 100%);
            z-index: 2;
        }

        /* Text on top of everything */
        .banner-text-overlay {
            position: relative;
            z-index: 10;
            text-align: center;
            padding: 0 20px;
            animation: fadeInUp 1s ease-out;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .banner-title {
            color: #ffffff;
            font-size: clamp(2.5rem, 6vw, 5rem);
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 6px;
            margin-bottom: 20px;
            text-shadow: 
                0 0 30px rgba(255, 140, 0, 0.6),
                0 0 60px rgba(255, 140, 0, 0.4),
                0 6px 40px rgba(0, 0, 0, 0.9);
            font-family: 'Segoe UI', 'Arial Black', sans-serif;
            background: linear-gradient(45deg, #fff, #ffb347, #ff8c00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 4px 15px rgba(0,0,0,0.7));
        }

        .banner-subtitle {
            color: rgba(255, 255, 255, 0.95);
            font-size: clamp(1.1rem, 2.5vw, 1.6rem);
            font-weight: 300;
            letter-spacing: 3px;
            text-shadow: 
                0 3px 20px rgba(0,0,0,0.9),
                0 0 15px rgba(255,255,255,0.2);
            font-family: 'Segoe UI', sans-serif;
        }

       /* Main Content */
        .main-content {
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
            position: relative;
            z-index: 10;
        }

        /* CARDS */
        .glass-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.5);
        }

        /* Control elements */
        .step-label { font-size: 1.1rem; font-weight: 700; color: #e3e3e3; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; display: block; }

        .generate-btn button {
            background: linear-gradient(135deg, #ff8c00 0%, #ff6b35 100%) !important;
            border: none !important; font-size: 1.3rem !important; font-weight: 700 !important; color: white !important;
            padding: 18px 50px !important; border-radius: 50px !important;
            box-shadow: 0 8px 25px rgba(255, 140, 0, 0.4) !important;
            width: 100%; transition: all 0.3s ease;
        }
        .generate-btn button:hover { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(255, 140, 0, 0.6) !important; }

        .image-container { border-radius: 15px; overflow: hidden; border: 2px solid rgba(0,0,0,0.05); }
        .prompt-box textarea { border: 2px solid #e0e0e0; border-radius: 12px; padding: 15px; }
        .prompt-box textarea:focus { border-color: #ff8c00; box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.1); }
        .ai-mode-radio input[type="radio"]:checked + label { background: linear-gradient(135deg, #ff8c00 0%, #ff6b35 100%); color: white; }

        footer { display: none !important; }
        """

        self.theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")

        with gr.Blocks(css=self.css, theme=self.theme, title='AI Interior Designer') as demo:
            # --- BANNER SECTION ---

            banner_src = self.encode_to_base64(Config.PATH_TO_BANNER)

            gr.HTML(f"""
            <div class="banner-container">
                <img src="{banner_src}" alt="Banner">
                <div class="banner-text-overlay">
                    <div class="banner-title">AI Interior Designer</div>
                    <div class="banner-subtitle">Redesign your space in seconds</div>
                </div>
            </div>
            """, elem_id="banner-component")

            # --- ONE COLUMN IN THE CENTER  ---
            with gr.Column(elem_classes="main-content"):
                original_image_state = gr.State(value=None)
                mask_state = gr.State(value=None)

                # --- STEP 1: INPUT PHOTO ---
                with gr.Group(elem_classes="glass-panel"):
                    gr.HTML('<span class="step-label" style="color: #ff6b35;">📤 Upload & Settings</span>')

                    # Large photo
                    input_image = gr.Image(
                        label="Input Image", show_label=False,
                        type='pil', interactive=True, height=550,
                        elem_classes="image-container"
                    )
                    # Mask clearing button
                    clear_btn = gr.Button("Clear Selected Area", variant="primary",
                                        elem_classes="generate-btn")
                    # Text window "User Guide"
                    gr.HTML("""
                            <div style="margin-top: 0px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; text-align: center;">
                                <p style="margin: 0; font-size: 0.95rem;">
                                    💡 <strong>Pro Tip:</strong> Click on objects in your image to select them, then describe your dream design!
                                </p>
                            </div>
                            """)
                    # Prompt and Buttons
                    with gr.Row():
                        with gr.Column(scale=3):
                            # Prompt
                            text_box = gr.Textbox(
                                label="Prompt", show_label=False, lines=3,
                                placeholder="Describe the new design here...",
                                elem_classes="prompt-box"
                            )
                        # Settings buttons
                        with gr.Column(scale=1):
                            control_select = gr.Radio(['depth', 'canny'], value='depth', label='Mode',
                                                      elem_classes="ai-mode-radio")

                    # Large button
                    run_btn = gr.Button("🎨 Generate Transformation", variant="primary", elem_classes="generate-btn")

                    with gr.Row():
                        # Examples catalog
                        gr.Examples(
                            examples=[
                                [Config.EXMP_1,
                                 "A luxurious, deep emerald green velvet upholstered ottoman bench with plush button tufting and dark wood legs, replacing the original beige bench.",
                                 "depth"],
                                [Config.EXMP_2,
                                 "A cozy, minimalist sofa upholstered in natural beige linen fabric, with relaxed, plush cushions. The sofa has light wood legs and is situated on the rug, casting soft shadows consistent with the room's lighting.",
                                 "depth"],
                                [Config.EXMP_3,
                                 "A complete grey fabric upholstered bed frame with neatly arranged white bedding and a dark grey knitted throw blanket. The entire base and wooden legs of the bed are clearly visible resting on the floor.",
                                 "depth"],
                            ],
                            inputs=[input_image, text_box, control_select],
                            label="Try these examples"
                        )
                # --- STEP 2: RESULT (Hidden) ---
                with gr.Group(visible=False, elem_classes="glass-panel") as results_group:
                    gr.HTML('<span class="step-label" style="color: #ff6b35;">✨ Transformation Result</span>')

                    output_image = gr.Image(
                        label="Result", show_label=False,
                        height=600,
                        elem_classes="image-container"
                    )

                # Logic
                input_image.upload(fn=self.on_upload, inputs=[input_image], outputs=[original_image_state, mask_state])
                input_image.select(fn=self.on_click, inputs=[original_image_state, mask_state],
                                   outputs=[input_image, mask_state])
                clear_btn.click(
                    fn=self.clear_mask,
                    inputs=[original_image_state, mask_state],
                    outputs=[input_image, mask_state]
                )
                run_btn.click(
                    fn=self.predict,
                    inputs=[original_image_state, text_box, control_select, mask_state],
                    outputs=[output_image, results_group]
                )

        return demo

