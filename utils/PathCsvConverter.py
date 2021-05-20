import csv

with open('log_415.csv', newline='') as File:
    reader = csv.reader(File)
    with open('log_415_path_edited.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for row in reader:
            print(row)
            values = row[0].split("/")
            row[0] = 'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\' + values[6] + "\\" + \
                     values[7] + "\\" + values[8] + "\\" + values[9]

            print(row[0])
            w.writerow([row[0], row[1], row[2]])
