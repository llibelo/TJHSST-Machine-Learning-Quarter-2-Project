import csv
import random
import sys

def split_csv(filename, train_filename, test_filename, train_ratio=0.8):
    with open(filename, 'r') as file:
        reader = list(csv.reader(file))
        headers = reader[0]
        rows = reader[1:]

    random.shuffle(rows)
    split_point = int(len(rows) * train_ratio)
    train_rows = rows[:split_point]
    test_rows = rows[split_point:]
    
    # Save training data
    with open(train_filename, 'w', newline='') as train_file:
        writer = csv.writer(train_file)
        writer.writerow(headers)
        writer.writerows(train_rows)
    
    # Save testing data
    with open(test_filename, 'w', newline='') as test_file:
        writer = csv.writer(test_file)
        writer.writerow(headers)
        writer.writerows(test_rows)
    
    print(f"Data split complete. Training data saved to '{train_filename}', Testing data saved to '{test_filename}'.")

# Example usage
split_csv(sys.argv[1], 'train.csv', 'test.csv', train_ratio=0.7)
