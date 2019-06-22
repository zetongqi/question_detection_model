import os
import urllib.request
import zipfile

DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE = os.path.join(DIR, "qnli.zip")
DOWNLOAD_URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601"
TRAIN = os.path.join(DIR, "QNLI", "train.tsv")
VAL = os.path.join(DIR, "QNLI", "dev.tsv")
TEST = os.path.join(DIR, "QNLI", "test.tsv")

if __name__ == '__main__':
	if not os.path.exists(ZIP_FILE):
		urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_FILE)
		zip_ref = zipfile.ZipFile(ZIP_FILE, 'r')
		zip_ref.extractall(DIR)
		zip_ref.close()
