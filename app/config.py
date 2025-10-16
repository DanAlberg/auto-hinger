# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the app directory with BOM-tolerant encoding
dotenv_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=dotenv_path, override=True, encoding="utf-8-sig")

# Main API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
