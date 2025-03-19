import argparse
from omegaconf import OmegaConf
from src.pipeline import get_pipeline

# Load configuration
CONFIG = OmegaConf.load('./conf.yaml')

# Parse command line arguments
parser = argparse.ArgumentParser()
choices = ['prepare-data', 'train-model', 'score-model']
parser.add_argument('-m', '--mode', dest='mode', choices=choices, help="Run Mode")
args = parser.parse_args()

# Run the appropriate pipeline
if args.mode:
    pipeline = get_pipeline(CONFIG, args.mode)
    pipeline.run()
else:
    parser.print_help()
