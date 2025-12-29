"""
WAG Training Data Extraction Script
====================================
Extracts headline and body copy training pairs from WAG History.xlsx

Usage:
    python extract_training_data.py --input "../../WAG History.xlsx" --output ./output/

Author: Enterprise Architecture Team
Created: November 2025
"""

import pandas as pd
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory for utils
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WAGDataExtractor:
    """Extracts and processes training data from WAG History file."""

    # Column mappings for WAG History
    COLUMN_MAPPING = {
        'EventName': 'event_name',
        'StartDate': 'start_date',
        'EndDate': 'end_date',
        'ItemCodeGroup01': 'wic_codes',
        'Headline': 'headline',
        'BodyCopy': 'body_copy',
        'Disclaimer': 'disclaimer',
        'AdRetail': 'ad_retail',
        'SingleItemPrice': 'single_item_price',
        'AdDollarOff': 'dollar_off',
        'PercentOff': 'percent_off',
        'FinalPrice': 'final_price',
        'LoPrice': 'lo_price',
        'HighPrice': 'high_price',
        'Limit': 'limit',
        'Quantity': 'quantity',
        'GetFree': 'get_free',
        'BOGOPercent': 'bogo_percent',
        'Pts': 'points',
        'Rewards': 'rewards'
    }

    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize the extractor.

        Args:
            input_path: Path to WAG History.xlsx
            output_dir: Directory for output files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.training_data: List[Dict] = []
        self.stats: Dict = {}

    def load_data(self) -> pd.DataFrame:
        """Load WAG History Excel file."""
        logger.info(f"Loading data from {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        # Read Excel file
        self.df = pd.read_excel(
            self.input_path,
            engine='openpyxl'
        )

        logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        logger.info(f"Columns: {list(self.df.columns)}")

        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the data."""
        logger.info("Cleaning data...")

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Store original count
        original_count = len(self.df)

        # Remove rows where both Headline and BodyCopy are empty
        self.df = self.df.dropna(subset=['Headline'], how='all')

        # Fill NaN values
        self.df['Headline'] = self.df['Headline'].fillna('')
        self.df['BodyCopy'] = self.df['BodyCopy'].fillna('')
        self.df['Disclaimer'] = self.df['Disclaimer'].fillna('')
        self.df['ItemCodeGroup01'] = self.df['ItemCodeGroup01'].fillna('')

        # Strip whitespace
        for col in ['Headline', 'BodyCopy', 'Disclaimer']:
            self.df[col] = self.df[col].astype(str).str.strip()

        # Remove rows with empty headlines after cleaning
        self.df = self.df[self.df['Headline'].str.len() > 0]

        cleaned_count = len(self.df)
        logger.info(f"Cleaned data: {original_count:,} -> {cleaned_count:,} records")
        logger.info(f"Removed {original_count - cleaned_count:,} records with empty headlines")

        return self.df

    def parse_wic_codes(self, wic_string: str) -> List[str]:
        """Parse comma-separated WIC codes into a list."""
        if pd.isna(wic_string) or not wic_string:
            return []

        # Split by comma and clean each code
        codes = str(wic_string).split(',')
        codes = [code.strip() for code in codes if code.strip()]

        return codes

    def safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert a value to float, handling strings and None."""
        if pd.isna(value) or value is None or value == '':
            return default
        try:
            # Handle string values that might have $ or other characters
            if isinstance(value, str):
                # Remove common currency symbols and whitespace
                cleaned = value.replace('$', '').replace(',', '').strip()
                if not cleaned:
                    return default
                return float(cleaned)
            return float(value)
        except (ValueError, TypeError):
            return default

    def determine_offer_type(self, row: pd.Series) -> str:
        """Determine the offer type from pricing columns."""
        offer_parts = []

        # Check for dollar off
        dollar_off = self.safe_float(row.get('AdDollarOff'))
        if dollar_off > 0:
            offer_parts.append(f"${dollar_off:.0f} Off")

        # Check for percent off
        percent_off = self.safe_float(row.get('PercentOff'))
        if percent_off > 0:
            offer_parts.append(f"{percent_off:.0f}% Off")

        # Check for BOGO - try both column name variations
        bogo_pct = self.safe_float(row.get('BOGOPercent', row.get('BogoPercentOff', 0)))
        if bogo_pct > 0:
            if bogo_pct == 100:
                offer_parts.append("Buy 1 Get 1 Free")
            else:
                offer_parts.append(f"BOGO {bogo_pct:.0f}% Off")

        # Check for Get Free
        get_free = self.safe_float(row.get('GetFree'))
        if get_free > 0:
            qty = self.safe_float(row.get('Quantity'), 1) or 1
            offer_parts.append(f"Buy {qty:.0f} Get {get_free:.0f} Free")

        # Check for rewards/points - try column name variations
        rewards = self.safe_float(row.get('Rewards', row.get('RewardValue', 0)))
        if rewards > 0:
            offer_parts.append(f"${rewards:.0f} Rewards")

        points = self.safe_float(row.get('Pts', row.get('RewardPoints', 0)))
        if points > 0:
            offer_parts.append(f"{points:.0f} Points")

        return ', '.join(offer_parts) if offer_parts else 'Regular Price'

    def format_price_info(self, row: pd.Series) -> str:
        """Format price information into a readable string."""
        parts = []

        # Ad retail price
        ad_retail = self.safe_float(row.get('AdRetail'))
        if ad_retail > 0:
            parts.append(f"${ad_retail:.2f}")

        # Price range
        lo = self.safe_float(row.get('LoPrice'))
        hi = self.safe_float(row.get('HighPrice'))
        if lo > 0 and hi > 0 and lo != hi:
            parts.append(f"${lo:.2f}-${hi:.2f}")

        # Final price
        final = self.safe_float(row.get('FinalPrice'))
        if final > 0:
            if parts:
                parts.append(f"Final: ${final:.2f}")
            else:
                parts.append(f"${final:.2f}")

        return ' | '.join(parts) if parts else 'Price varies'

    def create_training_example(self, row: pd.Series) -> Optional[Dict]:
        """Create a single training example from a row."""
        wic_codes = self.parse_wic_codes(row.get('ItemCodeGroup01', ''))
        headline = str(row.get('Headline', '')).strip()
        body_copy = str(row.get('BodyCopy', '')).strip()

        # Skip if no headline
        if not headline:
            return None

        # Build the training example
        example = {
            # Metadata
            'id': f"wag_{row.name}",
            'event_name': str(row.get('EventName', '')),
            'start_date': str(row.get('StartDate', '')),
            'end_date': str(row.get('EndDate', '')),

            # Input features
            'wic_codes': wic_codes,
            'wic_codes_str': ', '.join(wic_codes),
            'num_products': len(wic_codes),
            'price_info': self.format_price_info(row),
            'offer_type': self.determine_offer_type(row),
            'limit': str(row.get('Limit', '')) if pd.notna(row.get('Limit')) else '',

            # Target outputs
            'headline': headline,
            'body_copy': body_copy,
            'disclaimer': str(row.get('Disclaimer', '')).strip(),

            # Raw pricing data for enrichment
            'ad_retail': self.safe_float(row.get('AdRetail')),
            'final_price': self.safe_float(row.get('FinalPrice')),
            'dollar_off': self.safe_float(row.get('AdDollarOff')),
            'percent_off': self.safe_float(row.get('PercentOff')),
        }

        return example

    def extract_training_pairs(self) -> List[Dict]:
        """Extract all training pairs from the dataset."""
        logger.info("Extracting training pairs...")

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.training_data = []
        skipped = 0

        for idx, row in self.df.iterrows():
            example = self.create_training_example(row)
            if example:
                self.training_data.append(example)
            else:
                skipped += 1

        logger.info(f"Extracted {len(self.training_data):,} training pairs")
        logger.info(f"Skipped {skipped:,} records (empty headlines)")

        return self.training_data

    def compute_statistics(self) -> Dict:
        """Compute statistics about the extracted data."""
        logger.info("Computing statistics...")

        if not self.training_data:
            raise ValueError("No training data. Call extract_training_pairs() first.")

        headlines = [ex['headline'] for ex in self.training_data]
        body_copies = [ex['body_copy'] for ex in self.training_data]

        self.stats = {
            'total_records': len(self.training_data),
            'unique_headlines': len(set(headlines)),
            'unique_body_copies': len(set(body_copies)),
            'avg_headline_length': sum(len(h) for h in headlines) / len(headlines),
            'avg_body_copy_length': sum(len(b) for b in body_copies if b) / max(1, len([b for b in body_copies if b])),
            'records_with_body_copy': len([b for b in body_copies if b]),
            'records_with_wic_codes': len([ex for ex in self.training_data if ex['wic_codes']]),
            'avg_products_per_ad': sum(ex['num_products'] for ex in self.training_data) / len(self.training_data),
            'offer_type_distribution': {},
            'headline_length_distribution': {
                '0-20': 0,
                '21-40': 0,
                '41-60': 0,
                '61-80': 0,
                '80+': 0
            }
        }

        # Offer type distribution
        for ex in self.training_data:
            offer = ex['offer_type']
            self.stats['offer_type_distribution'][offer] = \
                self.stats['offer_type_distribution'].get(offer, 0) + 1

        # Headline length distribution
        for h in headlines:
            length = len(h)
            if length <= 20:
                self.stats['headline_length_distribution']['0-20'] += 1
            elif length <= 40:
                self.stats['headline_length_distribution']['21-40'] += 1
            elif length <= 60:
                self.stats['headline_length_distribution']['41-60'] += 1
            elif length <= 80:
                self.stats['headline_length_distribution']['61-80'] += 1
            else:
                self.stats['headline_length_distribution']['80+'] += 1

        return self.stats

    def create_instruction_format(self, example: Dict) -> Dict:
        """Convert example to instruction-tuning format."""
        # Build input prompt
        input_parts = [
            "Generate a headline and body copy for this retail advertisement.",
            "",
            f"WIC Codes: {example['wic_codes_str']}" if example['wic_codes_str'] else "",
            f"Number of Products: {example['num_products']}",
            f"Price: {example['price_info']}",
            f"Offer: {example['offer_type']}",
            f"Limit: {example['limit']}" if example['limit'] else "",
        ]

        input_text = '\n'.join([p for p in input_parts if p])

        # Build output
        output_parts = [f"Headline: {example['headline']}"]
        if example['body_copy']:
            output_parts.append(f"BodyCopy: {example['body_copy']}")

        output_text = '\n'.join(output_parts)

        return {
            'id': example['id'],
            'instruction': "You are a retail advertising copywriter. Generate concise, effective headlines and body copy for print advertisements.",
            'input': input_text,
            'output': output_text,
            'metadata': {
                'event_name': example['event_name'],
                'wic_codes': example['wic_codes'],
                'offer_type': example['offer_type']
            }
        }

    def save_outputs(self,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Dict[str, Path]:
        """
        Save training data in multiple formats with train/val/test splits.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing

        Returns:
            Dictionary of output file paths
        """
        logger.info("Saving outputs...")

        if not self.training_data:
            raise ValueError("No training data. Call extract_training_pairs() first.")

        # Shuffle data
        import random
        random.seed(42)
        shuffled_data = self.training_data.copy()
        random.shuffle(shuffled_data)

        # Split data
        n = len(shuffled_data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = shuffled_data[:train_end]
        val_data = shuffled_data[train_end:val_end]
        test_data = shuffled_data[val_end:]

        logger.info(f"Split: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")

        output_files = {}

        # Save raw extracted data (JSON)
        raw_path = self.output_dir / 'wag_training_data_raw.json'
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2, default=str)
        output_files['raw'] = raw_path
        logger.info(f"Saved raw data: {raw_path}")

        # Save instruction-format data (JSONL) for each split
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            jsonl_path = self.output_dir / f'wag_{split_name}.jsonl'
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for example in split_data:
                    instruction_example = self.create_instruction_format(example)
                    f.write(json.dumps(instruction_example, default=str) + '\n')
            output_files[split_name] = jsonl_path
            logger.info(f"Saved {split_name} data: {jsonl_path}")

        # Save statistics
        if self.stats:
            stats_path = self.output_dir / 'extraction_stats.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, default=str)
            output_files['stats'] = stats_path
            logger.info(f"Saved statistics: {stats_path}")

        # Save as CSV for easy viewing
        csv_path = self.output_dir / 'wag_training_data.csv'
        df_out = pd.DataFrame(self.training_data)
        df_out.to_csv(csv_path, index=False, encoding='utf-8')
        output_files['csv'] = csv_path
        logger.info(f"Saved CSV: {csv_path}")

        return output_files

    def run(self) -> Dict[str, Path]:
        """Run the complete extraction pipeline."""
        logger.info("=" * 60)
        logger.info("WAG Training Data Extraction Pipeline")
        logger.info("=" * 60)

        self.load_data()
        self.clean_data()
        self.extract_training_pairs()
        self.compute_statistics()
        output_files = self.save_outputs()

        # Print summary
        logger.info("=" * 60)
        logger.info("Extraction Complete!")
        logger.info("=" * 60)
        logger.info(f"Total training examples: {self.stats['total_records']:,}")
        logger.info(f"Unique headlines: {self.stats['unique_headlines']:,}")
        logger.info(f"Avg headline length: {self.stats['avg_headline_length']:.1f} chars")
        logger.info(f"Records with body copy: {self.stats['records_with_body_copy']:,}")
        logger.info("=" * 60)

        return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Extract training data from WAG History file'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../../WAG History.xlsx',
        help='Path to WAG History.xlsx'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../output/data',
        help='Output directory for extracted data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training data ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation data ratio (default: 0.1)'
    )

    args = parser.parse_args()

    # Calculate test ratio
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    # Run extraction
    extractor = WAGDataExtractor(args.input, args.output)
    output_files = extractor.run()

    print("\nOutput files created:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    main()
