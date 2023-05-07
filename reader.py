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
""" import csv

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Print header row
		header = next(reader)
		header_str = '|'.join([h.ljust(25) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(25) if r else '-69'.ljust(25) for r in row])
			print(row_str, file=outfile) """

""" import csv

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Print header row
		header = next(reader)
		header_str = '|'.join([h.ljust(10) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(10) if r else '-69'.ljust(10) for r in row])
			print(row_str, file=outfile) """

"""
para cada coluna sacar o numero de sim e nao para calcular a sua entropia.
"""
""" import csv

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Initialize counts for each result in each column
		counts = {}
		# Print header row
		header = next(reader)
		header_str = '|'.join([h.ljust(10) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(10) if r else '-69'.ljust(10) for r in row])
			print(row_str, file=outfile)
			# Update counts for each result in each column
			for i, val in enumerate(row):
				if header[i] not in counts:
					counts[header[i]] = {'RB': 0, 'NRB': 0}
				counts[header[i]][row[-1]] += 1
		# Print result counts for each column to the console
		for i, column in enumerate(header[:-1]):
			print(f"{column}:")
			rb_count = counts[column]['RB']
			nrb_count = counts[column]['NRB']
			print(f"\tRB: {rb_count}")
			print(f"\tNRB: {nrb_count}") """


""" import csv

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Initialize counts for each result in each column
		counts = {}
		# Print header row
		header = next(reader)
		header_str = '|'.join([h.ljust(10) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(10) if r else '-69'.ljust(10) for r in row])
			print(row_str, file=outfile)
			# Update counts for each result in each non-missing column
			for i, val in enumerate(row[:-1]):
				if val != '' and val != '-69':
					if header[i] not in counts:
						counts[header[i]] = {'RB': 0, 'NRB': 0}
					counts[header[i]][row[-1]] += 1
		# Print result counts for each column to the console
		for i, column in enumerate(header[:-1]):
			print(f"{column}:")
			if column in counts:
				rb_count = counts[column]['RB']
				nrb_count = counts[column]['NRB']
				print(f"\tRB: {rb_count}")
				print(f"\tNRB: {nrb_count}") """



""" import csv

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Initialize counts for each result in each column
		counts = {}
		# Print header row
		header = next(reader)
		header_str = '|'.join([h.ljust(30) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(30) if r else '-69'.ljust(30) for r in row])
			print(row_str, file=outfile)
			# Update counts for each result in each non-missing column
			for i, val in enumerate(row[:-1]):
				if val != '' and val != '-69':
					if header[i] not in counts:
						counts[header[i]] = {'RB': 0, 'NRB': 0}
					counts[header[i]][row[-1]] += 1
		# Print result counts for each column to the console
		max_rb_column = ''
		max_rb_count = 0
		for i, column in enumerate(header[:-1]):
			rb_count = counts.get(column, {'RB': 0})['RB']
			nrb_count = counts.get(column, {'NRB': 0})['NRB']
			print(f"{column}:")
			print(f"\tRB: {rb_count}")
			print(f"\tNRB: {nrb_count}")
			if rb_count > max_rb_count:
				max_rb_column = column
				max_rb_count = rb_count
		print(f"\nColumn with the biggest number of RB occurrences: {max_rb_column}")
		print(f"Number of RB occurrences: {max_rb_count}") """


""" import csv
import math

def calculate_class_entropy(class_labels):
	# Count the number of occurrences of each class label
	label_counts = {}
	for label in class_labels:
		if label not in label_counts:
			label_counts[label] = 0
		label_counts[label] += 1

	# Calculate the entropy of each class
	class_entropy = {}
	total_samples = len(class_labels)
	for label, count in label_counts.items():
		p = count / total_samples
		class_entropy[label] = -p * math.log2(p)

	return class_entropy

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Initialize counts for each result in each column
		counts = {}
		# Initialize list of class labels
		class_labels = []
		# Print header row
		header = next(reader)
		header_str = '|'.join([h.ljust(10) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(10) if r else '-69'.ljust(10) for r in row])
			print(row_str, file=outfile)
			# Update counts for each result in each non-missing column
			for i, val in enumerate(row[:-1]):
				if val != '' and val != '-69':
					if header[i] not in counts:
						counts[header[i]] = {'RB': 0, 'NRB': 0}
					counts[header[i]][row[-1]] += 1
			# Add class label to list
			class_labels.append(row[-1])
		# Print result counts for each column to the console
		max_rb_column = ''
		max_rb_count = 0
		for i, column in enumerate(header[:-1]):
			rb_count = counts.get(column, {'RB': 0})['RB']
			nrb_count = counts.get(column, {'NRB': 0})['NRB']
			print(f"{column}:")
			print(f"\tRB: {rb_count}")
			print(f"\tNRB: {nrb_count}")
			if rb_count > max_rb_count:
				max_rb_column = column
				max_rb_count = rb_count
		print(f"\nColumn with the biggest number of RB occurrences: {max_rb_column}")
		print(f"Number of RB occurrences: {max_rb_count}")

		# Calculate class entropy and print to console
		class_entropy = calculate_class_entropy(class_labels)
		print("\nClass entropy:")
		for label, entropy in class_entropy.items():
			print(f"\t{label}: {entropy}") """



""" import csv
import math

def calculate_class_entropy(class_labels):
	# Count the number of occurrences of each class label
	label_counts = {}
	for label in class_labels:
		if label not in label_counts:
			label_counts[label] = 0
		label_counts[label] += 1

	# Calculate the entropy of each class
	class_entropy = {}
	total_samples = len(class_labels)
	for label, count in label_counts.items():
		p = count / total_samples
		class_entropy[label] = -p * math.log2(p)

	return class_entropy

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Initialize counts for each result in each column
		counts = {}
		# Initialize list of class labels
		class_labels = []
		# Print header row
		header = next(reader)
		header_str = '|'.join([h.ljust(10) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(10) if r else '-69'.ljust(10) for r in row])
			print(row_str, file=outfile)
			# Update counts for each result in each non-missing column
			for i, val in enumerate(row[:-1]):
				if val != '' and val != '-69':
					if header[i] not in counts:
						counts[header[i]] = {'RB': 0, 'NRB': 0}
					counts[header[i]][row[-1]] += 1
			# Add class label to list
			class_labels.append(row[-1])
		# Print result counts for each column to the console
		max_rb_column = ''
		max_rb_count = 0
		for i, column in enumerate(header[:-1]):
			rb_count = counts.get(column, {'RB': 0})['RB']
			nrb_count = counts.get(column, {'NRB': 0})['NRB']
			print(f"{column}:")
			print(f"\tRB: {rb_count}")
			print(f"\tNRB: {nrb_count}")
			if rb_count > max_rb_count:
				max_rb_column = column
				max_rb_count = rb_count
		print(f"\nColumn with the biggest number of RB occurrences: {max_rb_column}")
		print(f"Number of RB occurrences: {max_rb_count}")

		# Calculate class entropy and print to console
		class_entropy = calculate_class_entropy(class_labels)
		print("\nClass entropy:")
		for label, entropy in class_entropy.items():
			print(f"\t{label}: {entropy}") """


""" import csv
import math

def calculate_class_entropy(class_labels):
	# Count the number of occurrences of each class label
	label_counts = {}
	for label in class_labels:
		if label not in label_counts:
			label_counts[label] = 0
		label_counts[label] += 1

	# Calculate the entropy of each class
	class_entropy = {}
	total_samples = len(class_labels)
	for label, count in label_counts.items():
		p = count / total_samples
		class_entropy[label] = -p * math.log2(p)

	return class_entropy

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Initialize counts for each result in each column
		counts = {}
		# Initialize list of class labels
		class_labels = []
		# Initialize list of column entropies
		column_entropies = [0] * (len(next(reader)) - 1)
		# Print header row
		header_str = '|'.join([h.ljust(10) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(10) if r else '-69'.ljust(10) for r in row])
			print(row_str, file=outfile)
			# Update counts for each result in each non-missing column
			for i, val in enumerate(row[:-1]):
				if val != '' and val != '-69':
					if header[i] not in counts:
						counts[header[i]] = {'RB': 0, 'NRB': 0}
					counts[header[i]][row[-1]] += 1
				# Update entropies for each column
				if val:
					if class_labels:
						class_labels.pop()
					class_labels.extend([row[i], row[-1]])
					column_entropies[i] += calculate_class_entropy(class_labels)[row[i]] * (counts[header[i]][row[-1]] / sum(counts[header[i]].values()))
			# Add class label to list
			class_labels.append(row[-1])
		# Print result counts for each column to the console
		max_rb_column = ''
		max_rb_count = 0
		for i, column in enumerate(header[:-1]):
			rb_count = counts.get(column, {'RB': 0})['RB']
			nrb_count = counts.get(column, {'NRB': 0})['NRB']
			print(f"{column}:")
			print(f"\tRB: {rb_count}")
			print(f"\tNRB: {nrb_count}")
			if rb_count > max_rb_count:
				max_rb_column = column
				max_rb_count = rb_count
		print(f"\nColumn with the biggest number of RB occurrences: {max_rb_column}")
		print(f"Number of RB occurrences: {max_rb_count}")

		# Print column entropies to console
		print("\nColumn entropies:")
		for i, column in enumerate(header[:-1]):
			print(f"{column}: {column_entropies[i]}") """

""" import csv
import math

def calculate_class_entropy(class_labels):
	# Count the number of occurrences of each class label
	label_counts = {}
	for label in class_labels:
		if label not in label_counts:
			label_counts[label] = 0
		label_counts[label] += 1

	# Calculate the entropy of each class
	class_entropy = {}
	total_samples = len(class_labels)
	for label, count in label_counts.items():
		p = count / total_samples
		class_entropy[label] = -p * math.log2(p)

	return class_entropy

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
	reader = csv.reader(file)
	# Initialize the header variable
	header = next(reader)
	# Open the output file for writing
	with open('prettier.txt', 'w') as outfile:
		# Initialize counts for each result in each column
		counts = {}
		# Initialize list of class labels
		class_labels = []
		# Initialize list of column entropies
		column_entropies = [0] * (len(header) - 1)
		# Print header row
		header_str = '|'.join([h.ljust(10) for h in header])
		print(header_str, file=outfile)
		print("-" * len(header_str), file=outfile)
		# Print data rows
		for row in reader:
			row_str = '|'.join([r.ljust(10) if r else '-69'.ljust(10) for r in row])
			print(row_str, file=outfile)
			# Update counts for each result in each non-missing column
			for i, val in enumerate(row[:-1]):
				if val != '' and val != '-69':
					if header[i] not in counts:
						counts[header[i]] = {'RB': 0, 'NRB': 0}
					counts[header[i]][row[-1]] += 1
				# Update entropies for each column
				if val:
					if class_labels:
						class_labels.pop()
					class_labels.extend([row[i], row[-1]])
					column_entropies[i] += calculate_class_entropy(class_labels)[row[i]] * (counts[header[i]][row[-1]] / sum(counts[header[i]].values()))
			# Add class label to list
			class_labels.append(row[-1])
		# Print result counts for each column to the console
		max_rb_column = ''
		max_rb_count = 0
		for i, column in enumerate(header[:-1]):
			rb_count = counts.get(column, {'RB': 0})['RB']
			nrb_count = counts.get(column, {'NRB': 0})['NRB']
			print(f"{column}:")
			print(f"\tRB: {rb_count}")
			print(f"\tNRB: {nrb_count}")
			if rb_count > max_rb_count:
				max_rb_column = column
				max_rb_count = rb_count
		print(f"\nColumn with the biggest number of RB occurrences: {max_rb_column}")
		print(f"Number of RB occurrences: {max_rb_count}")

		# Print column entropies to console
		print("\nColumn entropies:")
		for i, column in enumerate(header[:-1]):
			print(f"{column}: {column_entropies[i]}") """

""" import csv
import math

def calculate_class_entropy(class_labels):
    # Count the number of occurrences of each class label
    label_counts = {}
    for label in class_labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # Calculate the entropy of each class
    class_entropy = {}
    total_samples = len(class_labels)
    for label, count in label_counts.items():
        p = count / total_samples
        class_entropy[label] = -p * math.log2(p)

    return class_entropy

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
    reader = csv.reader(file)
    # Initialize the header variable
    header = next(reader)
    # Precompute the class entropies
    class_labels = []
    class_entropy = calculate_class_entropy(class_labels)
    # Initialize counts and entropies for each column
    column_info = {}
    for i, col in enumerate(header[:-1]):
        column_info[col] = {'counts': {'RB': 0, 'NRB': 0}, 'entropy': 0}
    # Open the output file for writing
    with open('prettier.txt', 'w') as outfile:
        # Print header row
        header_str = '|'.join([h.ljust(10) for h in header])
        print(header_str, file=outfile)
        print("-" * len(header_str), file=outfile)
        # Print data rows
        for row in reader:
            # Update counts for each result in each non-missing column
            for i, val in enumerate(row[:-1]):
                if val != '' and val != '-69':
                    column_info[header[i]]['counts'][row[-1]] += 1
            # Update entropies for each column
            for i, val in enumerate(row[:-1]):
                if val:
                    column_info[header[i]]['entropy'] += class_entropy.get(row[-1], 0) * (column_info[header[i]]['counts'][row[-1]] / sum(column_info[header[i]]['counts'].values()))
            # Add class label to list
            class_labels.append(row[-1])
        # Print result counts for each column to the console
        max_rb_column = ''
        max_rb_count = 0
        for i, column in enumerate(header[:-1]):
            rb_count = column_info[column]['counts'].get('RB', 0)
            nrb_count = column_info[column]['counts'].get('NRB', 0)
            print(f"{column}:")
            print(f"\tRB: {rb_count}")
            print(f"\tNRB: {nrb_count}")
            if rb_count > max_rb_count:
                max_rb_column = column
                max_rb_count = rb_count
        print(f"\nColumn with the biggest number of RB occurrences: {max_rb_column}")
        print(f"Number of RB occurrences: {max_rb_count}")

        # Print column entropies to console
        print("\nColumn entropies:")
        for i, column in enumerate(header[:-1]):
            print(f"{column}: {column_info[column]['entropy']}") """


import csv
import math

def calculate_class_entropy(class_labels):
    # Count the number of occurrences of each class label
    label_counts = {}
    for label in class_labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # Calculate the entropy of each class
    class_entropy = {}
    total_samples = len(class_labels)
    for label, count in label_counts.items():
        p = count / total_samples
        class_entropy[label] = -p * math.log2(p)

    return class_entropy

# Open the CSV file
with open('biodegradable_a.csv', 'r') as file:
    reader = csv.reader(file)
    # Initialize the header variable
    header = next(reader)
    # Initialize counts and entropies for each column
    column_info = {}
    for i, col in enumerate(header[:-1]):
        column_info[col] = {'counts': {'RB': 0, 'NRB': 0}, 'entropy': 0}
    # Open the output file for writing
    with open('prettier.txt', 'w') as outfile:
        # Print header row
        header_str = '|'.join([h.ljust(10) for h in header])
        print(header_str, file=outfile)
        print("-" * len(header_str), file=outfile)
        # Print data rows
        class_labels = []
        for row in reader:
            # Add class label to list
            class_labels.append(row[-1])
            # Update counts for each result in each non-missing column
            for i, val in enumerate(row[:-1]):
                if val != '' and val != '-69':
                    column_info[header[i]]['counts'][row[-1]] += 1
        # Precompute the class entropies
        class_entropy = calculate_class_entropy(class_labels)
        # Update entropies for each column
        for column in header[:-1]:
            total_count = sum(column_info[column]['counts'].values())
            for label in column_info[column]['counts']:
                p = column_info[column]['counts'][label] / total_count
                if p > 0:
                    column_info[column]['entropy'] -= p * math.log2(p)
        # Print result counts for each column to the console
        max_rb_column = ''
        max_rb_count = 0
        for i, column in enumerate(header[:-1]):
            rb_count = column_info[column]['counts'].get('RB', 0)
            nrb_count = column_info[column]['counts'].get('NRB', 0)
            print(f"{column}:")
            print(f"\tRB: {rb_count}")
            print(f"\tNRB: {nrb_count}")
            if rb_count > max_rb_count:
                max_rb_column = column
                max_rb_count = rb_count
        print(f"\nColumn with the biggest number of RB occurrences: {max_rb_column}")
        print(f"Number of RB occurrences: {max_rb_count}")

        # Print column entropies to console
        print("\nColumn entropies:")
        for i, column in enumerate(header[:-1]):
            print(f"{column}: {column_info[column]['entropy']}")





