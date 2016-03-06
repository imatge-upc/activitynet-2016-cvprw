import json

with open('../dataset/subdataset2.json') as raw_file:
    dataset = json.load(raw_file)

SUBSET = 'training'

with open('download_list_train.txt', 'w') as output_file:
    for key in dataset.keys():
        if dataset[key]['subset'] == SUBSET:
            output_file.write(key+'\n')
