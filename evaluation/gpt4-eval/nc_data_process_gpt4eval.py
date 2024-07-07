import re
import json
import logging
import os
import argparse
import pandas as pd


'''
python3 nc-data-process-gpt4eval.py --folder  /Users/karenq/Documents/code/gpt4-eval/data/nc/new
'''
# Configure logging
logging.basicConfig(level=logging.INFO)

def format_output_data(id, original_input, complete_assertion):
    return {
        "id": id,
        "instruction": original_input,
        "source_id": f"novacomet-{id}",
        "dataset": "NC-NL",
        "output": complete_assertion,
        "generator": "superNC",
        "datasplit": "NC"
    }

def clean_field(field):
    return field.replace('</s>', '').strip()

def extract_subfields(text):
    context = re.search(r'context: (.*?)(?: ;|$)', text)
    query = re.search(r'query: (.*?)(?: ;|$)', text)
    inference = re.search(r'inference: (.*?)(?:\n|$)', text)
    
    return {
        'context': context.group(1) if context else "",
        'query': query.group(1) if query else "",
        'inference': inference.group(1) if inference else ""
    }

def validate_assertion(context, query, inference):
    if not all([context, query, inference]):
        return False
    if len(context) < 3 or len(query) < 3 or len(inference) < 3:
        return False
    return True

def combine_fields(input_field, output_field):
    return clean_field(input_field.replace("__", "") + output_field.replace("__", ""))

def process_json(json_line, id):
    try:
        parsed_data = json.loads(json_line)
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON line: {json_line}")
        return None
    
    if 'input' not in parsed_data or 'output' not in parsed_data:
        logging.warning(f"Missing 'input' or 'output' in line: {json_line}")
        return None
    
    input_subfields = extract_subfields(parsed_data['input'])
    output_subfields = extract_subfields(parsed_data['output'])
    
    context = combine_fields(input_subfields['context'], output_subfields['context'])
    query = combine_fields(input_subfields['query'], output_subfields['query'])
    inference = combine_fields(input_subfields['inference'], output_subfields['inference'])
    
    if not validate_assertion(context, query, inference):
        logging.warning("Incomplete assertion found.")
    final_assertion = f"context: {context} ; query: {query} ; inference: {inference}"
    return format_output_data(id, parsed_data['input'], final_assertion)

def main(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, "processed_" + filename[:-4] + ".json")

            processed_data_complete = []

            try:
                df = pd.read_csv(input_path)
                for index, row in df.iterrows():
                    prompt = row['prompt']
                    output = row['output']

                    dataformat = format_output_data(index, prompt, output)
                    processed_data_complete.append(dataformat)
                    
            except FileNotFoundError:
                logging.error(f"File {input_path} not found")

            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data_complete, f)
            except Exception as e:
                logging.error(f"Could not write to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files in a folder.')
    parser.add_argument('--folder', required=True, help='Path to the folder containing JSON files.')
    parser.add_argument('--pretrained-model', default="llama", help='Path to the folder containing JSON files.')
    args = parser.parse_args()

    main(args.folder)
