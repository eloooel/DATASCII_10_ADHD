# DATASCII_10_ADHD

## Data Access

This project uses the ADHD-200 dataset. Due to size and licensing restrictions, the raw data is **not included** in this repository.

Please download the dataset from [ADHD-200 Preprocessed](http://fcon_1000.projects.nitrc.org/indi/adhd200/)

=========================================================================================

Instructions for Running:

Run preprocessing only:

python main.py --stage preprocessing

Run preprocessing + feature extraction:

python main.py --stage features

Run full pipeline (all stages):

python main.py --stage full

=========================================================================================

Instructions for downloading dependencies:

1. Input in the terminal
   pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

For Preprocessing and Feature Extraction:
Individual Subject-level parallel execution
