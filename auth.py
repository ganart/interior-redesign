import huggingface_hub
import logging
logging.basicConfig(level=logging.INFO)

"""
Utility script to authenticate with Hugging Face Hub.

Run this script once to log in and save your access token locally.
This allows the application to download models without entering credentials every time.
"""

logging.info("Starting the authorization process...")
huggingface_hub.login()
logging.info("Success! Token saved.")


