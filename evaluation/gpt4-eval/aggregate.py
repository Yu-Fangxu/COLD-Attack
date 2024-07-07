import os
import json
import csv

def aggregate_json_to_csv(folder_path, output_csv_file):
    json_data_list = []
    all_field_names = set()  # Use a set to store unique field names

    # Iterate over the files in the given folder path
    for file_name in os.listdir(folder_path):
        # Check if the file is a JSON file
        # /home/fangxu/gpt4_eval-main/outputs_selected/processed_guanaco.json-score_multi.gpt4.json
        if file_name.endswith('score_multi.gpt4.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                # Check if the data is a list of dictionaries
                if isinstance(data, list):
                    # Extend the list with the data items
                    json_data_list.extend(data)
                    # Update the set with keys from the first item, assuming all items have the same structure
                    all_field_names.update(data[0].keys())
                elif isinstance(data, dict):
                    # Append the item to the list
                    json_data_list.append(data)
                    # Update the set with keys from this item
                    all_field_names.update(data.keys())
                else:
                    # Handle other unexpected data formats
                    print(f"Error: Unexpected data format in file {file_name}")

    # Convert the set to a list and sort it to maintain consistency
    field_names = list(all_field_names)
    field_names.sort()

    with open(output_csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()
        for item in json_data_list:
            # Use the writerow method to write the item, missing fields will be empty
            csv_writer.writerow(item)

    print(f'All JSON data has been aggregated into {output_csv_file}')

# Use the function with your folder path and desired CSV output file name
folder_path = './outputs_selected'

output_csv_file = './outputs_selected/aggregated_data.csv'

aggregate_json_to_csv(folder_path, output_csv_file)
