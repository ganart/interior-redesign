import env_setup
from src.gradio_ui import WebUi
import logging

logging.basicConfig(
    level=logging.INFO,  # Уровень INFO, чтобы видеть ваши сообщения
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
def main():
    """Initialize and launch the AI Interior Designer Web UI."""
    ui = WebUi()

    # Create and launch the Gradio interface
    demo = ui.create_demo()
    demo.launch(inbrowser=True, share=False)

if __name__ == '__main__':
    main()