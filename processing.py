
# Tokenization
import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer

input_data_path = "/opt/ml/processing/input_data"
output_data_path = "/opt/ml/processing/output_data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="imdb")

    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    
    # tokenizer used in preprocessing
    tokenizer_name = args.tokenizer_name # 'distilbert-base-uncased'

    # dataset used
    dataset_name = args.dataset_name # 'imdb'

    # s3 key prefix for the data
#     s3_prefix = 'samples/datasets/imdb'

    # load dataset
#     dataset = load_dataset(dataset_name)

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    # load dataset
    train_dataset = load_dataset('csv', data_files={'train':os.path.join(input_data_path,'train.csv')})
    test_dataset = load_dataset('csv', data_files={'test':os.path.join(input_data_path,'test.csv')})
#     train_dataset, test_dataset = load_dataset(dataset_name, split=['train', 'test'])
#     test_dataset = test_dataset.shuffle().select(range(10000)) # smaller the size for test dataset to 10k 


    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


    # save dataset to /opt/ml/processing/
    train_dataset.save_to_disk(output_data_path)
    test_dataset.save_to_disk(output_data_path)
