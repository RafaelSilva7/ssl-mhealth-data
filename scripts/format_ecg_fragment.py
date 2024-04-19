import pandas as pd
import numpy as np
import wfdb
from pathlib import Path
import zipfile
import json
import os
import sys
import shutil


data_path = Path(sys.argv[1])
zip_filepath = Path(sys.argv[2])

records = {}

# ----------------------------------------------
# Read the dataset records
# ----------------------------------------------
with open(data_path / "RECORDS") as f:
    for x in f.readlines():
        record_path = x.replace("\n", '')
        key = record_path.split('/')[0]

        if key not in records.keys():
            records[key] = []

        records[key].append(record_path)


# ----------------------------------------------
# Split the dataset into train, val, and test
# Note: The class balance is maintained
# ----------------------------------------------
train_data, train_label = [], []
val_data, val_label = [], []
test_data, test_label = [], []

# split data by class
for key in records.keys():

    # read the class dir
    class_data = []
    for record in records[key]:
        class_data.append(wfdb.rdsamp(data_path / record)[0])

    class_data = np.stack(class_data)

    # split the dataset into - train, val and test 
    size_train_val, size_test = round(len(class_data)*.8), round(len(class_data)*.2)
    size_train, size_val = round(size_train_val*.8), round(size_train_val*.2)

    # train data
    range_train = np.arange(0, size_train)
    train_data.append(class_data[range_train])
    train_label.append([key]*len(range_train))

    # validation data
    range_val = np.arange(size_train, size_train+size_val)
    val_data.append(class_data[range_val])
    val_label.append([key]*len(range_val))

    # test data
    range_test = np.arange(range_val[-1]+1, (range_val[-1]+1)+size_test)
    test_data.append(class_data[range_test])
    test_label.append([key]*len(range_test))


# ----------------------------------------------
# Save samples as numpy and csv file
# ----------------------------------------------
temp_dir = Path("ecg-fragment")
os.makedirs(temp_dir, exist_ok=True)

train_data = np.concatenate(train_data)
train_data = np.moveaxis(train_data, -1, 1)
np.save(temp_dir / "train_samples.npy", train_data)

train_label = sum(train_label, [])
train_label = pd.DataFrame({'label': train_label})
train_label.to_csv(temp_dir / "train_labels.csv")


val_data = np.concatenate(val_data)
val_data = np.moveaxis(val_data, -1, 1)
np.save(temp_dir / "val_samples.npy", val_data)

val_label = sum(val_label, [])
val_label = pd.DataFrame({'label': val_label})
val_label.to_csv(temp_dir / "val_labels.csv")


test_data = np.concatenate(test_data)
test_data = np.moveaxis(test_data, -1, 1)
np.save(temp_dir / "test_samples.npy", test_data)

test_label = sum(test_label, [])
test_label = pd.DataFrame({'label': test_label})
test_label.to_csv(temp_dir / "test_labels.csv")


metadata = wfdb.rdsamp(data_path / records['1_Dangerous_VFL_VF'][0])[1]
with open(temp_dir / "metadata.json", 'x') as f:
    json.dump(metadata, f)


# ----------------------------------------------
# Zip the dataset files
# ----------------------------------------------
zf = zipfile.ZipFile(zip_filepath, 'w')
for dirname, subdir, files in os.walk(temp_dir.name):
    for filename in files:
        zf.write(temp_dir / filename, filename)

zf.close()

# ----------------------------------------------
# Remove the output dir
# ----------------------------------------------
shutil.rmtree(temp_dir)