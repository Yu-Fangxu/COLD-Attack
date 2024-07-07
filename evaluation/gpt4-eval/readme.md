# GPT-4 evaluation

### Prepare the data
Put the generated .csv to a folder ./eval, then run
```
python nc_data_process_gpt4eval.py
```
Then the data would be transformed to json format.

### Run the GPT-4 evaluation
```
bash run_attack.eval.sh [Your data path] [Your output path]
```
Note that you need to set the OPENAI_API_KEY with yours

### Aggregate the results
```
python aggregate.py
```
