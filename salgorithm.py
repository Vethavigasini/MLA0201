import csv
import os

def find_s(csv_path):
    hypothesis = None

    with open(csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header row

        for row in reader:
            if not row:
                continue

            attributes = [x.strip() for x in row[:-1]]
            label = row[-1].strip().lower()

            if label == 'yes':
                if hypothesis is None:
                    hypothesis = attributes.copy()
                else:
                    for i in range(len(hypothesis)):
                        if hypothesis[i] != attributes[i]:
                            hypothesis[i] = '?'

    return hypothesis


# ---- MAIN ----
filename = "PlayTennis.csv"

possible_paths = [
    filename,  # same folder as python file
    os.path.join(os.getcwd(), filename),
    os.path.join(os.path.expanduser("~"), "Downloads", filename),
    os.path.join(os.path.expanduser("~"), "Documents", filename),
    os.path.join(os.path.expanduser("~"), "Desktop", filename),
]

csv_path = None
for p in possible_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print("ERROR: PlayTennis.csv not found.")
    print("Put PlayTennis.csv in the same folder as this .py file OR in Downloads/Documents/Desktop.")
    print("Checked paths:")
    for p in possible_paths:
        print(" -", p)
else:
    print("Using dataset at:", csv_path)
    final_hypothesis = find_s(csv_path)
    print("Most Specific Hypothesis (FIND-S):")
    print(final_hypothesis)
