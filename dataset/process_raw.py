from qnli import TRAIN, VAL, TEST
import csv
import os

DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(DIR, "processed")
PROCESSED_TRAIN = os.path.join(PROCESSED, "train.tsv")
PROCESSED_VAL = os.path.join(PROCESSED, "val.tsv")
PROCESSED_TEST = os.path.join(PROCESSED, "test.tsv")


def process_raw_data(raw_path, processed_path):
	print("processing raw data file: ", raw_path)
	open(processed_path, "w").close()
	processed_tsv_train = csv.writer(open(processed_path, 'wt'), delimiter='\t')
	tsvTrain = csv.reader(open(raw_path), delimiter='\t')
	# skipping the header
	next(tsvTrain)
	for row in tsvTrain:
		processed_tsv_train.writerow([row[1], 1])
		processed_tsv_train.writerow([row[2], 0])


if __name__ == '__main__':
	if not os.path.exists(PROCESSED):
		# create processed data folder
		os.mkdir(PROCESSED)
		process_raw_data(TRAIN, PROCESSED_TRAIN)
		process_raw_data(VAL, PROCESSED_VAL)


