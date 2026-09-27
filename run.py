import sys
import os
import argparse

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

from pipeline import run_baseline_pipeline, run_training_and_validation, run_full_ml_inference

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    parser.add_argument('--baseline', action='store_true', help='Generate Submission 1 baseline')
    parser.add_argument('--train', action='store_true', help='Run ML training and threshold optimization')
    parser.add_argument('--test', action='store_true', help='Generate Submission 2 with full ML pipeline')
    parser.add_argument('--sample-size', type=int, default=35000, help='Training sample size')
    parser.add_argument('--threshold', type=float, default=0.80, help='Inference decision threshold')
    args = parser.parse_args()

    if args.baseline:
        run_baseline_pipeline()
    elif args.train:
        run_training_and_validation(sample_size=args.sample_size)
    elif args.test:
        run_full_ml_inference(threshold=args.threshold)
    else:
        print("Please specify --baseline, --train, or --test")

if __name__ == "__main__":
    main()
