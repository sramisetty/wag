"""
WAG Data Validation Script
===========================
Validates extracted training data quality and generates reports

Usage:
    python validate_data.py --input ../output/data/wag_training_data_raw.json

Author: Enterprise Architecture Team
Created: November 2025
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates training data quality and generates reports."""

    def __init__(self, data: List[Dict]):
        """Initialize validator with training data."""
        self.data = data
        self.issues: List[Dict] = []
        self.stats: Dict = {}

    def validate_headlines(self) -> Dict:
        """Validate headline quality."""
        logger.info("Validating headlines...")

        issues = []
        headline_lengths = []
        headline_patterns = Counter()

        for ex in self.data:
            headline = ex.get('headline', '')
            headline_lengths.append(len(headline))

            # Check for issues
            if len(headline) < 3:
                issues.append({
                    'id': ex.get('id'),
                    'issue': 'headline_too_short',
                    'value': headline
                })

            if len(headline) > 100:
                issues.append({
                    'id': ex.get('id'),
                    'issue': 'headline_too_long',
                    'value': headline[:50] + '...'
                })

            # Check for placeholder text
            if any(p in headline.lower() for p in ['xxx', 'tbd', 'placeholder', '???']):
                issues.append({
                    'id': ex.get('id'),
                    'issue': 'headline_placeholder',
                    'value': headline
                })

            # Pattern detection
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', headline):
                headline_patterns['Two Words Title Case'] += 1
            elif 'or' in headline.lower():
                headline_patterns['Multi-brand (contains "or")'] += 1
            elif headline.endswith('s'):
                headline_patterns['Ends with s (plural)'] += 1

        self.issues.extend(issues)

        return {
            'total_headlines': len(self.data),
            'avg_length': sum(headline_lengths) / len(headline_lengths),
            'min_length': min(headline_lengths),
            'max_length': max(headline_lengths),
            'issues_found': len(issues),
            'patterns': dict(headline_patterns.most_common(10))
        }

    def validate_body_copy(self) -> Dict:
        """Validate body copy quality."""
        logger.info("Validating body copy...")

        issues = []
        body_copy_patterns = Counter()
        has_body_copy = 0

        for ex in self.data:
            body_copy = ex.get('body_copy', '')

            if body_copy:
                has_body_copy += 1

                # Pattern detection
                if body_copy == 'Select varieties.':
                    body_copy_patterns['Select varieties.'] += 1
                elif 'Select varieties' in body_copy:
                    body_copy_patterns['Contains "Select varieties"'] += 1
                elif 'Limit' in body_copy:
                    body_copy_patterns['Contains "Limit"'] += 1
                elif re.match(r'^\d+', body_copy):
                    body_copy_patterns['Starts with number'] += 1
                else:
                    body_copy_patterns['Other'] += 1

                # Check for issues
                if len(body_copy) > 200:
                    issues.append({
                        'id': ex.get('id'),
                        'issue': 'body_copy_too_long',
                        'value': body_copy[:50] + '...'
                    })

        self.issues.extend(issues)

        return {
            'total_with_body_copy': has_body_copy,
            'total_without_body_copy': len(self.data) - has_body_copy,
            'coverage_rate': has_body_copy / len(self.data) * 100,
            'patterns': dict(body_copy_patterns.most_common(10)),
            'issues_found': len(issues)
        }

    def validate_wic_codes(self) -> Dict:
        """Validate WIC code data."""
        logger.info("Validating WIC codes...")

        issues = []
        wic_counts = []
        all_wics = set()

        for ex in self.data:
            wic_codes = ex.get('wic_codes', [])
            wic_counts.append(len(wic_codes))
            all_wics.update(wic_codes)

            # Check for issues
            if not wic_codes:
                issues.append({
                    'id': ex.get('id'),
                    'issue': 'no_wic_codes',
                    'value': ex.get('headline', '')
                })

            # Check for invalid WIC format
            for wic in wic_codes:
                if not re.match(r'^\d+$', str(wic)):
                    issues.append({
                        'id': ex.get('id'),
                        'issue': 'invalid_wic_format',
                        'value': wic
                    })

        self.issues.extend(issues)

        # Distribution of products per ad
        distribution = Counter(wic_counts)

        return {
            'total_unique_wics': len(all_wics),
            'avg_wics_per_example': sum(wic_counts) / len(wic_counts),
            'max_wics_per_example': max(wic_counts),
            'examples_without_wics': wic_counts.count(0),
            'wic_count_distribution': dict(sorted(distribution.items())[:10]),
            'issues_found': len(issues)
        }

    def validate_offer_types(self) -> Dict:
        """Validate and analyze offer types."""
        logger.info("Validating offer types...")

        offer_counts = Counter()

        for ex in self.data:
            offer_type = ex.get('offer_type', 'Unknown')
            offer_counts[offer_type] += 1

        return {
            'unique_offer_types': len(offer_counts),
            'distribution': dict(offer_counts.most_common(20))
        }

    def check_duplicates(self) -> Dict:
        """Check for duplicate entries."""
        logger.info("Checking for duplicates...")

        # Check headline duplicates
        headline_counts = Counter(ex.get('headline', '') for ex in self.data)
        duplicate_headlines = {h: c for h, c in headline_counts.items() if c > 1}

        # Check WIC combination duplicates
        wic_combo_counts = Counter(
            tuple(sorted(ex.get('wic_codes', [])))
            for ex in self.data
        )
        duplicate_wic_combos = sum(1 for c in wic_combo_counts.values() if c > 1)

        # Check exact duplicates (same headline + body copy)
        exact_combos = Counter(
            (ex.get('headline', ''), ex.get('body_copy', ''))
            for ex in self.data
        )
        exact_duplicates = sum(1 for c in exact_combos.values() if c > 1)

        return {
            'duplicate_headlines': len(duplicate_headlines),
            'most_common_headlines': list(headline_counts.most_common(10)),
            'duplicate_wic_combinations': duplicate_wic_combos,
            'exact_duplicates': exact_duplicates
        }

    def validate_for_training(self) -> Dict:
        """Check if data is suitable for LLM training."""
        logger.info("Validating training suitability...")

        suitable = 0
        unsuitable_reasons = Counter()

        for ex in self.data:
            is_suitable = True

            # Must have headline
            if not ex.get('headline'):
                is_suitable = False
                unsuitable_reasons['missing_headline'] += 1

            # Headline should be reasonable length
            headline_len = len(ex.get('headline', ''))
            if headline_len < 3 or headline_len > 100:
                is_suitable = False
                unsuitable_reasons['headline_length_issue'] += 1

            # Should have WIC codes or product info
            if not ex.get('wic_codes') and not ex.get('product_descriptions'):
                is_suitable = False
                unsuitable_reasons['no_product_info'] += 1

            if is_suitable:
                suitable += 1

        return {
            'suitable_for_training': suitable,
            'unsuitable_for_training': len(self.data) - suitable,
            'suitability_rate': suitable / len(self.data) * 100,
            'unsuitable_reasons': dict(unsuitable_reasons)
        }

    def run_all_validations(self) -> Dict:
        """Run all validation checks."""
        logger.info("=" * 60)
        logger.info("Running Data Validation")
        logger.info("=" * 60)

        self.stats = {
            'total_records': len(self.data),
            'headlines': self.validate_headlines(),
            'body_copy': self.validate_body_copy(),
            'wic_codes': self.validate_wic_codes(),
            'offer_types': self.validate_offer_types(),
            'duplicates': self.check_duplicates(),
            'training_suitability': self.validate_for_training(),
            'total_issues': len(self.issues)
        }

        return self.stats

    def generate_report(self, output_path: Path) -> Path:
        """Generate a validation report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_lines = [
            "# WAG Training Data Validation Report",
            f"\nGenerated: {__import__('datetime').datetime.now().isoformat()}",
            f"\n## Summary",
            f"- **Total Records:** {self.stats['total_records']:,}",
            f"- **Total Issues Found:** {self.stats['total_issues']:,}",
            f"- **Training Suitability:** {self.stats['training_suitability']['suitability_rate']:.1f}%",
            "",
            "## Headline Analysis",
            f"- Average Length: {self.stats['headlines']['avg_length']:.1f} characters",
            f"- Min Length: {self.stats['headlines']['min_length']}",
            f"- Max Length: {self.stats['headlines']['max_length']}",
            f"- Issues: {self.stats['headlines']['issues_found']}",
            "",
            "### Headline Patterns",
        ]

        for pattern, count in self.stats['headlines']['patterns'].items():
            report_lines.append(f"- {pattern}: {count:,}")

        report_lines.extend([
            "",
            "## Body Copy Analysis",
            f"- With Body Copy: {self.stats['body_copy']['total_with_body_copy']:,}",
            f"- Without Body Copy: {self.stats['body_copy']['total_without_body_copy']:,}",
            f"- Coverage Rate: {self.stats['body_copy']['coverage_rate']:.1f}%",
            "",
            "### Body Copy Patterns",
        ])

        for pattern, count in self.stats['body_copy']['patterns'].items():
            report_lines.append(f"- {pattern}: {count:,}")

        report_lines.extend([
            "",
            "## WIC Code Analysis",
            f"- Unique WICs: {self.stats['wic_codes']['total_unique_wics']:,}",
            f"- Avg WICs per Example: {self.stats['wic_codes']['avg_wics_per_example']:.1f}",
            f"- Max WICs per Example: {self.stats['wic_codes']['max_wics_per_example']}",
            f"- Examples Without WICs: {self.stats['wic_codes']['examples_without_wics']:,}",
            "",
            "## Offer Type Distribution",
        ])

        for offer, count in list(self.stats['offer_types']['distribution'].items())[:10]:
            report_lines.append(f"- {offer}: {count:,}")

        report_lines.extend([
            "",
            "## Duplicate Analysis",
            f"- Duplicate Headlines: {self.stats['duplicates']['duplicate_headlines']:,}",
            f"- Duplicate WIC Combinations: {self.stats['duplicates']['duplicate_wic_combinations']:,}",
            f"- Exact Duplicates: {self.stats['duplicates']['exact_duplicates']:,}",
            "",
            "## Training Suitability",
            f"- Suitable: {self.stats['training_suitability']['suitable_for_training']:,}",
            f"- Unsuitable: {self.stats['training_suitability']['unsuitable_for_training']:,}",
            "",
            "### Unsuitable Reasons",
        ])

        for reason, count in self.stats['training_suitability']['unsuitable_reasons'].items():
            report_lines.append(f"- {reason}: {count:,}")

        report = '\n'.join(report_lines)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Saved validation report: {output_path}")

        # Also save raw stats as JSON
        stats_path = output_path.with_suffix('.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)

        logger.info(f"Saved stats JSON: {stats_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Validate WAG training data quality'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../output/data/wag_training_data_raw.json',
        help='Path to training data JSON'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../output/reports/validation_report.md',
        help='Output path for validation report'
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data):,} records")

    # Run validation
    validator = DataValidator(data)
    stats = validator.run_all_validations()

    # Generate report
    report_path = validator.generate_report(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Records: {stats['total_records']:,}")
    print(f"Suitable for Training: {stats['training_suitability']['suitable_for_training']:,} ({stats['training_suitability']['suitability_rate']:.1f}%)")
    print(f"Total Issues: {stats['total_issues']:,}")
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
