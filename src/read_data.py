import csv

def read_data(file_name):
    with open(file_name, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)

    print(f"Loaded {len(data)} rows")
    print(f"Columns: {reader.fieldnames}")

    return data