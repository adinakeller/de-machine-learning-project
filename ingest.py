from datasets import load_dataset
import csv
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
    

def load_dataset(dataset, config):
    logger.info('loading dataset...')
    ds = load_dataset(dataset, config)
    train_data = ds['train']
    return train_data


def save_to_csv(file_name, data):
    with open(file_name, "w", encoding="utf-8") as file:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    logger.info(f'saved clean data to {file_name}')
    return file_name