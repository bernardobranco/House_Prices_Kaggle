import csv
with open('output_best.csv', 'r') as csvfile1:
    with open ("output_test_ensemble.csv", "r") as csvfile2:
        reader1 = csv.reader(csvfile1)
        reader2 = csv.reader(csvfile2)
        rows1_col_a = [row[0] for row in reader1]
        rows2 = [row for row in reader2]
        only_b = []
        for row in rows2:
            if row[0] not in rows1_col_a:
                only_b.append(row)
        print(only_b)