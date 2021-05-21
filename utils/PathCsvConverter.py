import csv


def windows_path_modification(data_training_number):
    data_training_path = "C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\training_data\\"
    with open(data_training_path+'log_' + str(data_training_number) + '.csv', newline='') as File:
        reader = csv.reader(File)
        with open(data_training_path+'log_' + str(data_training_number) + '_path_edited.csv', 'w', newline='') as f:
            w = csv.writer(f)
            for row in reader:
                print(row)
                values = row[0].split("/")
                row[0] = 'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\' + values[6] + "\\" \
                         + values[7] + "\\" + values[8] + "\\" + values[9]
                print(row[0])
                w.writerow([row[0], row[1], row[2]])
