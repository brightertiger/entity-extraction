import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.pipeline import get_pipeline


def main():
    config = OmegaConf.load(Path(__file__).parent / 'conf.yaml')
    
    parser = argparse.ArgumentParser(description='Entity Extraction Pipeline')
    choices = ['prepare-data', 'train-model', 'score-model']
    parser.add_argument('-m', '--mode', dest='mode', choices=choices, required=False, help='Run mode')
    args = parser.parse_args()

    if args.mode:
        pipeline = get_pipeline(config, args.mode)
        pipeline.run()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
