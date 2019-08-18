import os
import sys
import zipfile
import urllib.request

DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_LINK = (
    "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
)
BERT_ZIPFILE = os.path.join(DIR, "uncased_L-12_H-768_A-12.zip")
BERT_DIR = os.path.join(DIR, "uncased_L-12_H-768_A-12")
CHECKPOINT_FILE = os.path.join(BERT_DIR, "bert_model.ckpt")
CONFIG_JSON = os.path.join(BERT_DIR, "bert_config.json")
VOCAB_FILE = os.path.join(BERT_DIR, "vocab.txt")

if __name__ == "__main__":
    if not os.path.exists(BERT_ZIPFILE):
        print("downloading bert-base-uncased model")
        urllib.request.urlretrieve(DOWNLOAD_LINK, BERT_ZIPFILE)
        print("extracting zip file")
        with zipfile.ZipFile(BERT_ZIPFILE, "r") as zip_ref:
            zip_ref.extractall(DIR)
