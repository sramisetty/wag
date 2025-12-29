"""
WAG Ad Copy Generation Script
==============================
Generates headlines and body copy using the fine-tuned model

Usage:
    # Single generation
    python generate.py --wics "691500,691501" --price "$9.99" --offer "BOGO 50%"

    # Batch generation from EAB file
    python generate.py --eab ../../EABs/11.30\ EAB.xls --output generated_copy.json

    # Using Ollama API
    python generate.py --wics "691500" --use-ollama

Author: Enterprise Architecture Team
Created: November 2025
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of ad copy generation."""
    headline: str
    body_copy: str
    raw_output: str
    confidence: float
    metadata: Dict


class WAGGenerator:
    """Generates ad copy using fine-tuned model."""

    SYSTEM_PROMPT = """You are a retail advertising copywriter for Walgreens. Generate concise, effective headlines and body copy for print advertisements. Focus on clarity, brand consistency, and promotional messaging."""

    def __init__(self,
                 model_path: str = None,
                 use_ollama: bool = False,
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize generator.

        Args:
            model_path: Path to fine-tuned model (for local inference)
            use_ollama: Use Ollama API instead of local model
            ollama_url: URL of Ollama API
        """
        self.model_path = Path(model_path) if model_path else None
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.model = None
        self.tokenizer = None
        self.product_cache: Dict[str, Dict] = {}

    def load_local_model(self) -> None:
        """Load the fine-tuned model locally."""
        if not self.model_path:
            raise ValueError("Model path required for local inference")

        logger.info(f"Loading model from: {self.model_path}")

        # Load config
        config_path = self.model_path / "training_config.yaml"
        base_model = "mistralai/Mistral-7B-Instruct-v0.2"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            base_model = config['model']['name']

        # Quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load adapter
        adapter_path = self.model_path / "adapter"
        if adapter_path.exists():
            self.model = PeftModel.from_pretrained(base, adapter_path)
        else:
            self.model = PeftModel.from_pretrained(base, self.model_path)

        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def load_product_data(self, master_item_path: str) -> None:
        """Load product data for WIC lookups."""
        try:
            import pandas as pd

            logger.info(f"Loading product data from: {master_item_path}")
            df = pd.read_excel(master_item_path, engine='openpyxl', dtype={'WIC': str})

            for _, row in df.iterrows():
                wic = str(row.get('WIC', '')).strip()
                if wic:
                    self.product_cache[wic] = {
                        'description': str(row.get('Description', '')).strip(),
                        'brand': str(row.get('Brand', '')).strip(),
                        'vendor': str(row.get('Vendor', '')).strip(),
                    }

            logger.info(f"Loaded {len(self.product_cache)} products")

        except Exception as e:
            logger.warning(f"Could not load product data: {e}")

    def get_product_info(self, wic: str) -> Optional[Dict]:
        """Get product information by WIC."""
        return self.product_cache.get(str(wic).strip())

    def build_prompt(self,
                     wic_codes: List[str] = None,
                     products: List[Dict] = None,
                     price: str = None,
                     offer: str = None,
                     limit: str = None) -> str:
        """Build generation prompt."""
        # Get product info if WICs provided
        if wic_codes and not products:
            products = []
            for wic in wic_codes:
                info = self.get_product_info(wic)
                if info:
                    products.append(info)

        # Build input section
        input_parts = ["Generate a headline and body copy for this retail advertisement.", ""]

        if products:
            input_parts.append("Products:")
            for p in products[:5]:
                desc = p.get('description', 'Unknown product')
                brand = p.get('brand', '')
                if brand:
                    input_parts.append(f"  - {desc} (Brand: {brand})")
                else:
                    input_parts.append(f"  - {desc}")
            input_parts.append("")

        if price:
            input_parts.append(f"Price: {price}")
        if offer:
            input_parts.append(f"Offer: {offer}")
        if limit:
            input_parts.append(f"Limit: {limit}")

        input_text = '\n'.join(input_parts)

        # Format full prompt
        prompt = f"""### Instruction:
{self.SYSTEM_PROMPT}

### Input:
{input_text}

### Response:
"""
        return prompt

    def generate_local(self,
                       prompt: str,
                       max_new_tokens: int = 100,
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> str:
        """Generate using local model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_local_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "### Response:" in generated:
            response = generated.split("### Response:")[-1].strip()
        else:
            response = generated[len(prompt):].strip()

        return response

    def generate_ollama(self,
                        prompt: str,
                        model: str = "wag-copywriter",
                        temperature: float = 0.7) -> str:
        """Generate using Ollama API."""
        url = f"{self.ollama_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def parse_output(self, text: str) -> Tuple[str, str]:
        """Parse generated text into headline and body copy."""
        import re

        headline = ""
        body_copy = ""

        # Extract headline
        headline_match = re.search(r'Headline:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if headline_match:
            headline = headline_match.group(1).strip()

        # Extract body copy
        body_match = re.search(r'BodyCopy:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if body_match:
            body_copy = body_match.group(1).strip()

        # Fallback: treat first line as headline
        if not headline and text:
            lines = text.strip().split('\n')
            headline = lines[0].strip()
            # Remove "Headline:" prefix if present
            headline = re.sub(r'^Headline:\s*', '', headline, flags=re.IGNORECASE)
            if len(lines) > 1:
                body_copy = lines[1].strip()
                body_copy = re.sub(r'^BodyCopy:\s*', '', body_copy, flags=re.IGNORECASE)

        return headline, body_copy

    def generate(self,
                 wic_codes: List[str] = None,
                 products: List[Dict] = None,
                 price: str = None,
                 offer: str = None,
                 limit: str = None,
                 temperature: float = 0.7) -> GenerationResult:
        """
        Generate ad copy.

        Args:
            wic_codes: List of WIC codes
            products: List of product dicts (alternative to wic_codes)
            price: Price string (e.g., "$9.99")
            offer: Offer description (e.g., "BOGO 50%")
            limit: Purchase limit (e.g., "2")
            temperature: Generation temperature

        Returns:
            GenerationResult with headline, body copy, and metadata
        """
        # Build prompt
        prompt = self.build_prompt(
            wic_codes=wic_codes,
            products=products,
            price=price,
            offer=offer,
            limit=limit
        )

        # Generate
        if self.use_ollama:
            raw_output = self.generate_ollama(prompt, temperature=temperature)
        else:
            if self.model is None:
                self.load_local_model()
            raw_output = self.generate_local(prompt, temperature=temperature)

        # Parse output
        headline, body_copy = self.parse_output(raw_output)

        return GenerationResult(
            headline=headline,
            body_copy=body_copy,
            raw_output=raw_output,
            confidence=1.0,  # Could be enhanced with actual confidence scoring
            metadata={
                'wic_codes': wic_codes,
                'price': price,
                'offer': offer,
            }
        )

    def generate_batch(self,
                       items: List[Dict],
                       temperature: float = 0.7) -> List[GenerationResult]:
        """Generate ad copy for multiple items."""
        results = []

        for item in items:
            result = self.generate(
                wic_codes=item.get('wic_codes'),
                products=item.get('products'),
                price=item.get('price'),
                offer=item.get('offer'),
                limit=item.get('limit'),
                temperature=temperature
            )
            results.append(result)

        return results


class EABProcessor:
    """Processes EAB files for batch generation."""

    def __init__(self, generator: WAGGenerator):
        self.generator = generator

    def process_eab(self, eab_path: str, history_path: str = None) -> List[Dict]:
        """
        Process an EAB file and generate copy for new items.

        Args:
            eab_path: Path to EAB Excel file
            history_path: Path to WAG History for matching existing copy

        Returns:
            List of generated results
        """
        import pandas as pd

        logger.info(f"Processing EAB: {eab_path}")

        # Load EAB
        df = pd.read_excel(eab_path, engine='xlrd' if eab_path.endswith('.xls') else 'openpyxl')

        logger.info(f"Loaded {len(df)} rows from EAB")

        # Group by Page and Layout to collect WIC sets
        groups = df.groupby(['Page', 'Layout'])

        results = []

        for (page, layout), group in groups:
            # Collect WIC codes for this ad slot
            wic_codes = group['WIC'].dropna().astype(str).tolist()

            if not wic_codes:
                continue

            # Get pricing info (use first row)
            first_row = group.iloc[0]
            price = f"${first_row.get('Ad Retail', 0):.2f}" if pd.notna(first_row.get('Ad Retail')) else None

            # Generate copy
            result = self.generator.generate(
                wic_codes=wic_codes,
                price=price
            )

            results.append({
                'page': page,
                'layout': layout,
                'wic_codes': wic_codes,
                'headline': result.headline,
                'body_copy': result.body_copy,
                'price': price,
            })

        logger.info(f"Generated copy for {len(results)} ad slots")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate ad copy using fine-tuned WAG model'
    )

    # Generation mode
    parser.add_argument('--wics', type=str, help='Comma-separated WIC codes')
    parser.add_argument('--price', type=str, help='Price (e.g., "$9.99")')
    parser.add_argument('--offer', type=str, help='Offer description (e.g., "BOGO 50%")')
    parser.add_argument('--limit', type=str, help='Purchase limit')

    # Batch mode
    parser.add_argument('--eab', type=str, help='Path to EAB file for batch processing')

    # Model settings
    parser.add_argument('--model', '-m', type=str, default='../output/models/wag-copywriter',
                        help='Path to fine-tuned model')
    parser.add_argument('--use-ollama', action='store_true', help='Use Ollama API')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Ollama API URL')
    parser.add_argument('--ollama-model', type=str, default='wag-copywriter',
                        help='Ollama model name')

    # Product data
    parser.add_argument('--master-item', type=str, default='../../WAG Master Item.xlsx',
                        help='Path to WAG Master Item.xlsx')

    # Output
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')

    args = parser.parse_args()

    # Initialize generator
    generator = WAGGenerator(
        model_path=args.model if not args.use_ollama else None,
        use_ollama=args.use_ollama,
        ollama_url=args.ollama_url
    )

    # Load product data
    if Path(args.master_item).exists():
        generator.load_product_data(args.master_item)

    # Process based on mode
    if args.eab:
        # Batch mode
        processor = EABProcessor(generator)
        results = processor.process_eab(args.eab)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            for r in results:
                print(f"\nPage {r['page']}, Layout {r['layout']}:")
                print(f"  Headline: {r['headline']}")
                print(f"  Body: {r['body_copy']}")

    elif args.wics:
        # Single generation
        wic_codes = [w.strip() for w in args.wics.split(',')]

        result = generator.generate(
            wic_codes=wic_codes,
            price=args.price,
            offer=args.offer,
            limit=args.limit,
            temperature=args.temperature
        )

        print("\n" + "=" * 60)
        print("GENERATED AD COPY")
        print("=" * 60)
        print(f"Headline: {result.headline}")
        print(f"Body Copy: {result.body_copy}")
        print("=" * 60)

        if args.output:
            output = {
                'wic_codes': wic_codes,
                'headline': result.headline,
                'body_copy': result.body_copy,
                'metadata': result.metadata
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
