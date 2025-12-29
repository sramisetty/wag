"""
WAG Model Evaluation Script
============================
Evaluates fine-tuned model performance on test data

Usage:
    python evaluate.py --model ../output/models/wag-copywriter --test ../output/data/wag_test.jsonl

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
from typing import Dict, List, Tuple
from collections import defaultdict
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WAGEvaluator:
    """Evaluates fine-tuned WAG copywriter model."""

    def __init__(self, model_path: str, base_model: str = None):
        """
        Initialize evaluator.

        Args:
            model_path: Path to fine-tuned adapter
            base_model: Base model name (optional, reads from config)
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.results: List[Dict] = []

    def load_model(self) -> None:
        """Load the fine-tuned model."""
        # Try to load config for base model name
        config_path = self.model_path / "training_config.yaml"
        if config_path.exists() and not self.base_model:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.base_model = config['model']['name']

        if not self.base_model:
            self.base_model = "mistralai/Mistral-7B-Instruct-v0.2"

        logger.info(f"Loading base model: {self.base_model}")
        logger.info(f"Loading adapter from: {self.model_path}")

        # Load in 4-bit for inference
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load adapter
        adapter_path = self.model_path / "adapter"
        if adapter_path.exists():
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """Generate text from prompt."""
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

        # Decode and extract only the new tokens
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the response part (after "### Response:")
        if "### Response:" in generated:
            response = generated.split("### Response:")[-1].strip()
        else:
            # Try to find the generated part by removing the input
            response = generated[len(prompt):].strip()

        return response

    def format_prompt(self, example: Dict) -> str:
        """Format example into prompt."""
        return f"""### Instruction:
{example.get('instruction', '')}

### Input:
{example.get('input', '')}

### Response:
"""

    def parse_output(self, text: str) -> Tuple[str, str]:
        """Parse generated text into headline and body copy."""
        headline = ""
        body_copy = ""

        # Try to extract headline
        headline_match = re.search(r'Headline:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if headline_match:
            headline = headline_match.group(1).strip()

        # Try to extract body copy
        body_match = re.search(r'BodyCopy:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if body_match:
            body_copy = body_match.group(1).strip()

        # If no structured output, treat first line as headline
        if not headline and text:
            lines = text.strip().split('\n')
            headline = lines[0].strip()
            if len(lines) > 1:
                body_copy = lines[1].strip()

        return headline, body_copy

    def compute_exact_match(self, pred: str, ref: str) -> float:
        """Compute exact match score."""
        return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0

    def compute_partial_match(self, pred: str, ref: str) -> float:
        """Compute partial match score based on word overlap."""
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())

        if not ref_words:
            return 1.0 if not pred_words else 0.0

        overlap = len(pred_words & ref_words)
        precision = overlap / len(pred_words) if pred_words else 0
        recall = overlap / len(ref_words) if ref_words else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def compute_length_similarity(self, pred: str, ref: str) -> float:
        """Compute length similarity score."""
        pred_len = len(pred)
        ref_len = len(ref)

        if ref_len == 0:
            return 1.0 if pred_len == 0 else 0.0

        ratio = min(pred_len, ref_len) / max(pred_len, ref_len)
        return ratio

    def evaluate_example(self, example: Dict) -> Dict:
        """Evaluate a single example."""
        # Format prompt
        prompt = self.format_prompt(example)

        # Generate
        generated = self.generate(prompt)

        # Parse outputs
        pred_headline, pred_body = self.parse_output(generated)

        # Parse reference
        ref_output = example.get('output', '')
        ref_headline, ref_body = self.parse_output(ref_output)

        # Compute metrics
        result = {
            'id': example.get('id', ''),
            'input': example.get('input', ''),
            'reference_headline': ref_headline,
            'reference_body_copy': ref_body,
            'predicted_headline': pred_headline,
            'predicted_body_copy': pred_body,
            'raw_generated': generated,
            'metrics': {
                'headline_exact_match': self.compute_exact_match(pred_headline, ref_headline),
                'headline_partial_match': self.compute_partial_match(pred_headline, ref_headline),
                'headline_length_sim': self.compute_length_similarity(pred_headline, ref_headline),
                'body_exact_match': self.compute_exact_match(pred_body, ref_body),
                'body_partial_match': self.compute_partial_match(pred_body, ref_body),
            }
        }

        return result

    def evaluate_dataset(self,
                        test_file: str,
                        num_samples: int = None,
                        output_file: str = None) -> Dict:
        """Evaluate on test dataset."""
        logger.info(f"Loading test data from: {test_file}")

        # Load test data
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"Evaluating on {len(test_data)} samples...")

        # Evaluate each example
        self.results = []
        for example in tqdm(test_data, desc="Evaluating"):
            result = self.evaluate_example(example)
            self.results.append(result)

        # Aggregate metrics
        aggregated = self.aggregate_metrics()

        # Save results
        if output_file:
            self.save_results(output_file, aggregated)

        return aggregated

    def aggregate_metrics(self) -> Dict:
        """Aggregate metrics across all results."""
        metrics = defaultdict(list)

        for result in self.results:
            for metric_name, value in result['metrics'].items():
                metrics[metric_name].append(value)

        aggregated = {
            'num_samples': len(self.results),
            'metrics': {}
        }

        for metric_name, values in metrics.items():
            aggregated['metrics'][metric_name] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
            }

        # Add summary scores
        aggregated['metrics']['headline_score'] = (
            aggregated['metrics']['headline_exact_match']['mean'] * 0.5 +
            aggregated['metrics']['headline_partial_match']['mean'] * 0.5
        )
        aggregated['metrics']['body_score'] = (
            aggregated['metrics']['body_exact_match']['mean'] * 0.5 +
            aggregated['metrics']['body_partial_match']['mean'] * 0.5
        )
        aggregated['metrics']['overall_score'] = (
            aggregated['metrics']['headline_score'] * 0.6 +
            aggregated['metrics']['body_score'] * 0.4
        )

        return aggregated

    def save_results(self, output_file: str, aggregated: Dict) -> None:
        """Save evaluation results."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_path = output_path.with_suffix('.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'aggregated': aggregated,
                'detailed': self.results
            }, f, indent=2, default=str)

        logger.info(f"Saved detailed results: {results_path}")

        # Generate markdown report
        report_path = output_path.with_suffix('.md')
        self.generate_report(report_path, aggregated)
        logger.info(f"Saved report: {report_path}")

    def generate_report(self, report_path: Path, aggregated: Dict) -> None:
        """Generate evaluation report."""
        lines = [
            "# WAG Model Evaluation Report",
            "",
            f"**Model:** {self.model_path}",
            f"**Samples Evaluated:** {aggregated['num_samples']}",
            "",
            "## Summary Scores",
            "",
            f"| Metric | Score |",
            f"|--------|-------|",
            f"| Overall Score | {aggregated['metrics']['overall_score']:.3f} |",
            f"| Headline Score | {aggregated['metrics']['headline_score']:.3f} |",
            f"| Body Copy Score | {aggregated['metrics']['body_score']:.3f} |",
            "",
            "## Detailed Metrics",
            "",
            "### Headline Metrics",
            "",
            f"| Metric | Mean | Min | Max |",
            f"|--------|------|-----|-----|",
        ]

        for metric in ['headline_exact_match', 'headline_partial_match', 'headline_length_sim']:
            m = aggregated['metrics'][metric]
            lines.append(f"| {metric} | {m['mean']:.3f} | {m['min']:.3f} | {m['max']:.3f} |")

        lines.extend([
            "",
            "### Body Copy Metrics",
            "",
            f"| Metric | Mean | Min | Max |",
            f"|--------|------|-----|-----|",
        ])

        for metric in ['body_exact_match', 'body_partial_match']:
            m = aggregated['metrics'][metric]
            lines.append(f"| {metric} | {m['mean']:.3f} | {m['min']:.3f} | {m['max']:.3f} |")

        lines.extend([
            "",
            "## Sample Outputs",
            "",
        ])

        # Add sample outputs
        for i, result in enumerate(self.results[:5]):
            lines.extend([
                f"### Sample {i+1}",
                "",
                f"**Input:** {result['input'][:200]}...",
                "",
                f"**Reference Headline:** {result['reference_headline']}",
                f"**Predicted Headline:** {result['predicted_headline']}",
                "",
                f"**Reference Body:** {result['reference_body_copy']}",
                f"**Predicted Body:** {result['predicted_body_copy']}",
                "",
                "---",
                "",
            ])

        report = '\n'.join(lines)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned WAG copywriter model'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to fine-tuned model/adapter'
    )
    parser.add_argument(
        '--test', '-t',
        type=str,
        default='../output/data/wag_test.jsonl',
        help='Path to test data JSONL file'
    )
    parser.add_argument(
        '--base-model', '-b',
        type=str,
        default=None,
        help='Base model name (if not in config)'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../output/reports/evaluation',
        help='Output path for results'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = WAGEvaluator(args.model, args.base_model)

    # Load model
    evaluator.load_model()

    # Run evaluation
    results = evaluator.evaluate_dataset(
        args.test,
        num_samples=args.num_samples,
        output_file=args.output
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples: {results['num_samples']}")
    print(f"Overall Score: {results['metrics']['overall_score']:.3f}")
    print(f"Headline Score: {results['metrics']['headline_score']:.3f}")
    print(f"Body Copy Score: {results['metrics']['body_score']:.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
