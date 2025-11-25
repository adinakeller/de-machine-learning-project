from datasets import load_dataset
import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
    

def load(dataset, config):
    logger.info('loading dataset...')
    ds = load_dataset(dataset, config)
    all_data = ds['train']
    return all_data

def save_to_csv(file_name, data):
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    return df
