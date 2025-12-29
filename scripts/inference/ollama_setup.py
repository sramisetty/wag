"""
Ollama Setup and Export Script
==============================
Converts fine-tuned model to Ollama-compatible format

Usage:
    python ollama_setup.py --model ../output/models/wag-copywriter --export

Author: Enterprise Architecture Team
Created: November 2025
"""

import os
import sys
import json
import shutil
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        logger.info(f"Ollama version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("Ollama is not installed. Please install from https://ollama.ai")
        return False


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    import requests
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False


def export_to_gguf(model_path: Path, output_path: Path) -> Path:
    """
    Export model to GGUF format.

    Note: This requires llama.cpp to be installed.
    """
    logger.info("Exporting model to GGUF format...")
    logger.info("This requires llama.cpp conversion tools.")

    # Check for conversion script
    convert_script = Path("convert-lora-to-ggml.py")

    if not convert_script.exists():
        logger.warning("""
GGUF conversion requires llama.cpp tools. To set up:

1. Clone llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp

2. Install requirements:
   pip install -r llama.cpp/requirements.txt

3. Run conversion:
   python llama.cpp/convert-lora-to-ggml.py {model_path}

4. Quantize (optional):
   ./llama.cpp/quantize {output}.gguf {output}-q4_k_m.gguf q4_k_m

For now, you can use the model via the Python inference script instead.
""")
        return None

    # Run conversion
    output_file = output_path / "wag-copywriter-adapter.gguf"

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_file)
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Exported GGUF to: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        return None


def create_ollama_model(modelfile_path: Path, model_name: str = "wag-copywriter") -> bool:
    """Create Ollama model from Modelfile."""
    if not check_ollama_installed():
        return False

    if not check_ollama_running():
        logger.error("Ollama server is not running. Start with: ollama serve")
        return False

    logger.info(f"Creating Ollama model: {model_name}")

    cmd = ['ollama', 'create', model_name, '-f', str(modelfile_path)]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Model '{model_name}' created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create model: {e}")
        return False


def test_ollama_model(model_name: str = "wag-copywriter") -> bool:
    """Test the Ollama model with a sample prompt."""
    import requests

    prompt = """Generate a headline and body copy for this retail advertisement.

Products:
  - ADVIL PM 20CT (Brand: Advil)

Price: $8.99
Offer: BOGO 50%"""

    logger.info("Testing model with sample prompt...")

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': prompt,
                'stream': False,
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json().get('response', '')
            logger.info(f"Model response:\n{result}")
            return True
        else:
            logger.error(f"API error: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False


def list_ollama_models() -> list:
    """List available Ollama models."""
    import requests

    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
    except:
        pass

    return []


def setup_without_finetuning(model_name: str = "wag-copywriter-base"):
    """
    Set up a base model with system prompt for immediate use.

    This creates a model that uses the base Mistral with the WAG system prompt,
    allowing immediate use before fine-tuning is complete.
    """
    logger.info("Setting up base model with WAG system prompt...")

    modelfile_content = '''FROM mistral:7b-instruct

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM """You are a retail advertising copywriter for Walgreens. Generate concise, effective headlines and body copy for print advertisements.

Based on the products, pricing, and promotions provided, create:
1. A clear, brand-focused headline (typically the brand or product category name)
2. Brief body copy (usually "Select varieties." with any purchase limits)

Examples of good outputs:
- Headline: "Advil Pain Relief" / BodyCopy: "Select varieties."
- Headline: "Cold & Flu Relief" / BodyCopy: "Select varieties. Limit 2."
- Headline: "Bausch + Lomb or Blink Eye Care" / BodyCopy: "Select varieties."

Always format your response as:
Headline: [headline]
BodyCopy: [body copy]"""
'''

    # Write temporary Modelfile
    temp_modelfile = Path("Modelfile.temp")
    with open(temp_modelfile, 'w') as f:
        f.write(modelfile_content)

    try:
        success = create_ollama_model(temp_modelfile, model_name)
        if success:
            logger.info(f"""
Base model setup complete!

To use immediately:
    ollama run {model_name}

Or via API:
    curl http://localhost:11434/api/generate -d '{{
        "model": "{model_name}",
        "prompt": "Products: ADVIL PM 20CT | Price: $8.99 | Offer: BOGO 50%",
        "stream": false
    }}'

This base model will work reasonably well. After fine-tuning,
create the fine-tuned model with:
    python ollama_setup.py --model ../output/models/wag-copywriter --create
""")
    finally:
        if temp_modelfile.exists():
            temp_modelfile.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Setup Ollama for WAG copywriter model'
    )
    parser.add_argument('--model', '-m', type=str,
                        default='../output/models/wag-copywriter',
                        help='Path to fine-tuned model')
    parser.add_argument('--export', action='store_true',
                        help='Export model to GGUF format')
    parser.add_argument('--create', action='store_true',
                        help='Create Ollama model from Modelfile')
    parser.add_argument('--test', action='store_true',
                        help='Test the Ollama model')
    parser.add_argument('--list', action='store_true',
                        help='List available Ollama models')
    parser.add_argument('--setup-base', action='store_true',
                        help='Setup base model for immediate use (no fine-tuning required)')
    parser.add_argument('--name', type=str, default='wag-copywriter',
                        help='Ollama model name')

    args = parser.parse_args()

    # List models
    if args.list:
        models = list_ollama_models()
        print("\nAvailable Ollama models:")
        for m in models:
            print(f"  - {m}")
        return

    # Setup base model
    if args.setup_base:
        setup_without_finetuning(f"{args.name}-base")
        return

    # Export to GGUF
    if args.export:
        model_path = Path(args.model)
        output_path = model_path.parent
        export_to_gguf(model_path / "adapter", output_path)

    # Create Ollama model
    if args.create:
        modelfile_path = Path(__file__).parent / "Modelfile"
        create_ollama_model(modelfile_path, args.name)

    # Test model
    if args.test:
        test_ollama_model(args.name)

    # If no action specified, show help
    if not any([args.export, args.create, args.test, args.list, args.setup_base]):
        parser.print_help()
        print("\nQuick start:")
        print("  1. Setup base model: python ollama_setup.py --setup-base")
        print("  2. After training:   python ollama_setup.py --create --test")


if __name__ == '__main__':
    main()
