import json 
from collections import defaultdict
file = "example_data/eval_outputs/all.tag.gpt-4.json"
file_supp = "example_data/eval_outputs/all.tag.gpt-4.supp.json"
with open(file) as f:
    data = json.load(f)
with open(file_supp) as f:
    data_supp = json.load(f)
  
    
tags = []
tags_list = []

 

task_types = ["math", "coding", "writing", "info-seek", "role-play", "procedure", "reasoning"]
topics = ["stem", "humanities", "lifestyle", "finance", "medical", "nature", "ethics", "malicious"]
difficulty = ["1", "2", "3", "4", "5"]

# set default to be 0 
task_counts = defaultdict(int)
topic_counts = defaultdict(int)
difficulty_counts = defaultdict(int)



for item, item_sup in zip(data, data_supp):
    tag_item = {"id": item["id"], "instruction": item["input"]}
    tags_list_item = {"id": item["id"], "instruction": item["input"], "task_types": [], "topics": [], "difficulty": "n/a"}
    for task in task_types:
        tag_item[task] = task in item_sup["result"]
        if tag_item[task]:
            tags_list_item["task_types"].append(task)
        task_counts[task] += 1 if tag_item[task] else 0                    
    for topic in topics:
        tag_item[topic] = topic in item["result"]
        if tag_item[topic]:
            tags_list_item["topics"].append(topic)
        topic_counts[topic] += 1 if tag_item[topic] else 0
    for diff in difficulty:
        tag_item[diff] = f": {diff}" in item_sup["result"] or f", {diff}" in item_sup["result"]
        if tag_item[diff]:
            if 0 < int(diff) <=2 :
                tags_list_item["difficulty"] = "easy"
            elif 2 < int(diff) <= 3:
                tags_list_item["difficulty"] = "medium"
            elif int(diff) >= 4:
                tags_list_item["difficulty"] = "hard"
        difficulty_counts[diff] += 1 if tag_item[diff] else 0
    tags.append(tag_item)
    tags_list.append(tags_list_item)

print("Task Counts")
print(json.dumps(task_counts, indent=4))
print("Topic Counts")
print(json.dumps(topic_counts, indent=4))
print("Difficulty Counts")
print(json.dumps(difficulty_counts, indent=4))


with open("example_data/eval_outputs/tags.json", "w") as f:
    json.dump(tags, f, indent=4)

with open("example_data/eval_outputs/tags_list.json", "w") as f:
    json.dump(tags_list, f, indent=2)