"""
WAG Product Enrichment Script
==============================
Enriches training data with product details from WAG Master Item.xlsx

Usage:
    python enrich_with_products.py --training ../output/data/wag_training_data_raw.json --master "../../WAG Master Item.xlsx"

Author: Enterprise Architecture Team
Created: November 2025
"""

import pandas as pd
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductEnricher:
    """Enriches training data with product information from Master Item file."""

    def __init__(self, master_item_path: str):
        """
        Initialize the enricher.

        Args:
            master_item_path: Path to WAG Master Item.xlsx
        """
        self.master_item_path = Path(master_item_path)
        self.product_lookup: Dict[str, Dict] = {}
        self.brand_lookup: Dict[str, Set[str]] = defaultdict(set)
        self.category_lookup: Dict[str, Set[str]] = defaultdict(set)

    def load_master_item(self) -> pd.DataFrame:
        """Load and index the master item file."""
        logger.info(f"Loading master item file: {self.master_item_path}")

        if not self.master_item_path.exists():
            raise FileNotFoundError(f"Master item file not found: {self.master_item_path}")

        # Read Excel - this may take a while for 888K records
        df = pd.read_excel(
            self.master_item_path,
            engine='openpyxl',
            dtype={'WIC': str, 'UPC': str}
        )

        logger.info(f"Loaded {len(df):,} products")

        # Build lookup dictionary
        logger.info("Building product lookup index...")
        for _, row in df.iterrows():
            wic = str(row.get('WIC', '')).strip()
            if wic:
                self.product_lookup[wic] = {
                    'wic': wic,
                    'upc': str(row.get('UPC', '')).strip(),
                    'description': str(row.get('Description', '')).strip(),
                    'brand': str(row.get('Brand', '')).strip(),
                    'vendor': str(row.get('Vendor', '')).strip(),
                    'category_code': str(row.get('Prod Cat Code', '')).strip(),
                    'category_manager': str(row.get('CM', '')).strip(),
                }

                # Build reverse lookups
                brand = self.product_lookup[wic]['brand']
                if brand:
                    self.brand_lookup[brand].add(wic)

                category = self.product_lookup[wic]['category_code']
                if category:
                    self.category_lookup[category].add(wic)

        logger.info(f"Indexed {len(self.product_lookup):,} products")
        logger.info(f"Found {len(self.brand_lookup):,} unique brands")
        logger.info(f"Found {len(self.category_lookup):,} unique categories")

        return df

    def get_product_info(self, wic: str) -> Optional[Dict]:
        """Get product information by WIC code."""
        return self.product_lookup.get(str(wic).strip())

    def enrich_example(self, example: Dict) -> Dict:
        """Enrich a single training example with product details."""
        enriched = example.copy()

        wic_codes = example.get('wic_codes', [])
        if not wic_codes:
            enriched['products'] = []
            enriched['brands'] = []
            enriched['categories'] = []
            enriched['product_descriptions'] = []
            return enriched

        # Lookup each WIC
        products = []
        brands = set()
        categories = set()
        descriptions = []
        vendors = set()

        for wic in wic_codes:
            product = self.get_product_info(wic)
            if product:
                products.append(product)
                if product['brand']:
                    brands.add(product['brand'])
                if product['category_code']:
                    categories.add(product['category_code'])
                if product['description']:
                    descriptions.append(product['description'])
                if product['vendor']:
                    vendors.add(product['vendor'])

        enriched['products'] = products
        enriched['brands'] = list(brands)
        enriched['categories'] = list(categories)
        enriched['vendors'] = list(vendors)
        enriched['product_descriptions'] = descriptions
        enriched['products_found'] = len(products)
        enriched['products_missing'] = len(wic_codes) - len(products)

        return enriched

    def enrich_training_data(self, training_data: List[Dict]) -> List[Dict]:
        """Enrich all training examples."""
        logger.info(f"Enriching {len(training_data):,} training examples...")

        enriched_data = []
        found_count = 0
        missing_count = 0

        for i, example in enumerate(training_data):
            enriched = self.enrich_example(example)
            enriched_data.append(enriched)

            found_count += enriched.get('products_found', 0)
            missing_count += enriched.get('products_missing', 0)

            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1:,} examples...")

        logger.info(f"Enrichment complete!")
        logger.info(f"Products found: {found_count:,}")
        logger.info(f"Products missing: {missing_count:,}")
        logger.info(f"Match rate: {found_count / (found_count + missing_count) * 100:.1f}%")

        return enriched_data

    def create_enhanced_instruction_format(self, example: Dict) -> Dict:
        """Create instruction format with enriched product details."""
        # Build detailed input prompt
        input_parts = [
            "Generate a headline and body copy for this retail advertisement.",
            ""
        ]

        # Add product details if available
        if example.get('product_descriptions'):
            input_parts.append("Products:")
            for desc in example['product_descriptions'][:5]:  # Limit to 5 products
                input_parts.append(f"  - {desc}")
            input_parts.append("")

        # Add brand info
        if example.get('brands'):
            brands_str = ', '.join(example['brands'][:3])  # Limit to 3 brands
            input_parts.append(f"Brands: {brands_str}")

        # Add pricing and offer
        input_parts.append(f"Price: {example.get('price_info', 'Price varies')}")
        input_parts.append(f"Offer: {example.get('offer_type', 'Regular Price')}")

        if example.get('limit'):
            input_parts.append(f"Limit: {example['limit']}")

        input_text = '\n'.join(input_parts)

        # Build output
        output_parts = [f"Headline: {example['headline']}"]
        if example.get('body_copy'):
            output_parts.append(f"BodyCopy: {example['body_copy']}")

        output_text = '\n'.join(output_parts)

        return {
            'id': example.get('id', ''),
            'instruction': "You are a retail advertising copywriter for Walgreens. Generate concise, effective headlines and body copy for print advertisements based on the product and promotion details provided.",
            'input': input_text,
            'output': output_text,
            'metadata': {
                'wic_codes': example.get('wic_codes', []),
                'brands': example.get('brands', []),
                'offer_type': example.get('offer_type', ''),
                'products_found': example.get('products_found', 0)
            }
        }

    def save_enriched_data(self,
                          enriched_data: List[Dict],
                          output_dir: Path,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1) -> Dict[str, Path]:
        """Save enriched training data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        # Save full enriched data
        full_path = output_dir / 'wag_enriched_full.json'
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, default=str)
        output_files['full'] = full_path
        logger.info(f"Saved full enriched data: {full_path}")

        # Shuffle and split
        import random
        random.seed(42)
        shuffled = enriched_data.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }

        # Save instruction-format JSONL for each split
        for split_name, split_data in splits.items():
            jsonl_path = output_dir / f'wag_enriched_{split_name}.jsonl'
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for example in split_data:
                    instruction_example = self.create_enhanced_instruction_format(example)
                    f.write(json.dumps(instruction_example, default=str) + '\n')
            output_files[split_name] = jsonl_path
            logger.info(f"Saved {split_name}: {jsonl_path} ({len(split_data):,} examples)")

        return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Enrich training data with product details'
    )
    parser.add_argument(
        '--training', '-t',
        type=str,
        default='../output/data/wag_training_data_raw.json',
        help='Path to extracted training data JSON'
    )
    parser.add_argument(
        '--master', '-m',
        type=str,
        default='../../WAG Master Item.xlsx',
        help='Path to WAG Master Item.xlsx'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../output/data_enriched',
        help='Output directory for enriched data'
    )

    args = parser.parse_args()

    # Load master item data
    enricher = ProductEnricher(args.master)
    enricher.load_master_item()

    # Load training data
    logger.info(f"Loading training data from {args.training}")
    with open(args.training, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    logger.info(f"Loaded {len(training_data):,} training examples")

    # Enrich data
    enriched_data = enricher.enrich_training_data(training_data)

    # Save outputs
    output_files = enricher.save_enriched_data(enriched_data, args.output)

    print("\nOutput files created:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    main()
