import csv


with open('log_269.csv', newline='') as File:  
    reader = csv.reader(File)
    with open('../ia/training_data/log_269.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for row in reader:
            print(row)
            values = row[0].split("/")
            line = 'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\' + values[6] +"\\" + values [7] + "\\" + values[8] + "\\" +  values[9]
            row[0] = line
            print(row[0])
            #print(line)
            w.writerow([row[0], row[1], row[2]])
