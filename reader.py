""" import csv

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
    reader = csv.reader(file)
    # Print header row
    header = next(reader)
    print("|".join(header))
    print("-" * 50)
    # Print data rows
    for row in reader:
        print("|".join(row))
 """
import csv

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
    reader = csv.reader(file)
    # Open the output file for writing
    with open('prettier.txt', 'w') as outfile:
        # Print header row
        header = next(reader)
        header_str = '|'.join([h.ljust(20) for h in header])
        print(header_str, file=outfile)
        print("-" * len(header_str), file=outfile)
        # Print data rows
        for row in reader:
            row_str = '|'.join([r.ljust(20) if r else '-69'.ljust(20) for r in row])
            print(row_str, file=outfile)
