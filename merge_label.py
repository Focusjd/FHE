import os

with open('./tox21_data/ranged_data/training_data.txt', 'r') as data_file, open('./tox21_data/ranged_data/y_train.txt', 'r') as label_file:

    data_lines = data_file.readlines()
    label_lines = label_file.readlines()
    
    if len(data_lines) != len(label_lines):
        print("Error: The number of lines in training_data.txt and y_train.txt are not the same.")
        exit(1)

    with open('./tox21_data/ranged_data/train_labeled.txt', 'w') as output_file:
        for data, label in zip(data_lines, label_lines):
            combined_line = data.strip() + " " + label
            output_file.write(combined_line)

print("Merging completed.")