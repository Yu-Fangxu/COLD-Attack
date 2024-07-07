import argparse
import os
import json
import openai 
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from .utils import (
    better_json_loads,
    retry_handler, 
    openai_chat_request,
    PAIRWISE_TEMPLATE,
    # SCORE_TEMPLATE, 
    # SCORE_REF_TEMPLATE,
    # MULTI_SCORE_TEMPLATE,
    SAFETY_SCORE_TEMPLATE,
    # MULTI_SCORE_REF_TEMPLATE,
    # MULTI_PAIRWISE_TEMPLATE,
    TAG_DATA_TEMPLATE,
    TAG_DATA_TEMPLATE_V2,
    MULTI_PAIRWISE_TEMPLATE_nc,
    MULTI_SCORE_TEMPLATE_nc,
    MULTI_SCORE_TEMPLATE_planning,
    MULTI_SCORE_TEMPLATE_conditional_planning,
    MULTI_SCORE_TEMPLATE_counterfactual_planning,
    MULTI_SCORE_TEMPLATE_dimSum,
    MULTI_SCORE_TEMPLATE_dimPara,
    MULTI_SCORE_TEMPLATE_Soda,
    ASR_TEMPLATE
)
import numpy as np 
 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_only', action='store_true')
    
    parser.add_argument("--mode", type=str, default="pairwise", required=True)
    parser.add_argument("--subtask", type=str, default="planning")
    parser.add_argument("--first_file", type=str, required=False)
    parser.add_argument("--second_file", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1) 
    parser.add_argument("--reference_file", type=str, required=False) 
    parser.add_argument("--save_interval", type=int, default=3)
    
    # Prompt configs 
    parser.add_argument("--max_words_to_eval", type=int, default=-1)
    
    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-0314")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    args = parser.parse_args() 
    if args.api_key is not None:
        openai.api_key = args.api_key 
    
    if args.report_only:
        print("\nloading:", args.output_file)
        # assert not os.path.exists(args.output_file) 
        
    return args

def report(results, mode, args):

    if mode.startswith("pairwise"):
        cnt = 0
        same_cnt = 0
        eval_res = {}
        lens = {}
        for item in results:
            if "parsed_result" in item:
                d = item
                d["len_A"] = len(d["output_A"].split())
                d["len_B"] = len(d["output_B"].split())
                if mode == "pairwise":
                    label = item["parsed_result"]["preference"]
                    if label.upper() in ["A", "B", "SAME"]:
                        cnt += 1 
                        if label.upper() == "SAME":
                            same_cnt += 1
                            eval_res["same"] += 1
                        else:
                            winner = item[f"generator_{label}"]
                            eval_res[winner] = eval_res.get(winner, 0) + 1
                elif mode == "pairwise_multi":
                    for aspect, aspect_result in item["parsed_result"].items():
                        label = aspect_result["preference"]
                        if aspect not in eval_res:
                            eval_res[aspect] = {"same": 0}
                        if label.upper() in ["A", "B", "SAME"]:
                            cnt += 1 
                        if label.upper() == "SAME":
                            same_cnt += 1
                            eval_res[aspect]["same"] += 1
                        else:
                            winner = item[f"generator_{label}"]
                            eval_res[aspect][winner] = eval_res[aspect].get(winner, 0) + 1
                            
                for l in ["A", "B"]:
                    m = item[f"generator_{l}"]
                    if m not in lens:
                        lens[m] = []
                    lens[m].append(item["len_"+l]) 
                        
        eval_res.update({"total": cnt})
        eval_res["avg_lens"] = {}
        for m in lens:
            eval_res["avg_lens"][m] = float(np.mean(lens[m]))
            
    elif mode.startswith("score"):
        cnt = 0
        lens_cand = []
        if "_multi" not in mode and "_safety" not in mode:
            scores = []
        else:
            scores = {}
        if "+ref" in mode:
            lens_ref = []
        
        eval_res = {}
        
        for item in results:
            if "parsed_result" in item:
                d = item
                d["len_cand"] = len(d["output_cand"].split())
                lens_cand.append(item["len_cand"])
                if "_multi" not in mode and "_safety" not in mode:
                    score = item["parsed_result"]["score"]
                    scores.append(float(score))
                else:
                    for aspect, result in item["parsed_result"].items():
                        if aspect not in scores:
                            scores[aspect] = []
                        if result["score"] == "N/A":
                            result["score"] = 10.0
                        scores[aspect].append(float(result["score"]))
                if mode.endswith("+ref"):
                    d["len_ref"] = len(d["output_ref"].split())
                    lens_ref.append(item["len_ref"])
                
                cnt += 1
        if "_multi" not in mode and "_safety" not in mode:
            eval_res = {"total": cnt, "average_score": float(np.mean(scores)), "std": float(np.std(scores))}
        else:
            eval_res = {"total": cnt}
            for aspect, score_list in scores.items():
                eval_res[aspect + "_mean"] = float(np.mean(score_list))
                # eval_res[aspect + "_std"] = float(np.std(score_list))
                
        if "+ref" in mode: 
            eval_res["avg_lens"] = {"cand": float(np.mean(lens_cand)), "ref": float(np.mean(lens_ref))}
        else:
            eval_res["avg_lens"] = float(np.mean(lens_cand))
    
    elif mode.startswith("reward"):
        cnt = 0
        lens_cand = [] 
        scores = []  
        eval_res = {} 
        for item in results: 
            d = item
            d["len_cand"] = len(d["output_cand"].split())
            lens_cand.append(item["len_cand"]) 
            score = item["result"]["score"]
            scores.append(float(score))              
            cnt += 1 
        eval_res = {"total": cnt, "average_score": float(np.mean(scores)), "std": float(np.std(scores))} 
        eval_res["avg_lens"] = float(np.mean(lens_cand))
    eval_res["output_file"] = args.output_file
    return eval_res
                
def gpt_eval(results, args):
    # try to load the existing results from args.output_file 
    if os.path.exists(args.output_file):
        cnt = 0 
        with open(args.output_file, "r") as f:
            existing_results = json.load(f) 
        for i in range(len(existing_results)):
            e = existing_results[i]
            t = results[i+args.start_idx]
            if e["prompt"] != t["prompt"]:
                continue
            # if e["prompt"] == t["prompt"] and e["result"] != "N/A":
            #     results[i]["result"] = e["result"]
            #     cnt += 1 
            if "result" in e:
                t["result"] = e["result"]
                if "parsed_result" in e: 
                    t["parsed_result"] = e["parsed_result"]
                cnt += 1
        print(f"loading {cnt} results from {args.output_file}")
    openai_args = {
        "prompt": "TODO",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": []
    }
    if args.model:
        openai_args['model'] = args.model
    if args.engine:
        openai_args['engine'] = args.engine
        
    @retry_handler(retry_limit=10)
    def api(ind, item, **kwargs):
        result = openai_chat_request(**kwargs)
        result = result[0]
        if args.mode == "tag":
            return result 
        
        result = result.replace("```", "")
        if '\\\\"' in result:
            result = result.replace('\\\\"', '\\"')
        else:
            result = result.replace("\\", "\\\\")
        result = result.strip()
        if result[0] != "{" or result[0] != "}":
            start_index = result.find("{")
            end_index = result.rfind("}") + 1
            result = result[start_index:end_index]
        
        
        try:
            # json.loads(result)
            better_json_loads(result)
        except Exception as e:
            print(ind)
            print(e)
            print(result)
            raise e
        return result
    
    results = results[args.start_idx:args.end_idx] # for debug
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.output_file} "):
        if item["result"] != "N/A":
            if args.mode != "tag":
                results[ind]["parsed_result"] = better_json_loads(results[ind]["result"])
            print(f"Skipping {ind} for {args.output_file}")
            continue
        # print(f"\nNot Skipping {ind}") 
    
        openai_args["prompt"] = item["prompt"]
        try:
            result = api(ind, item, **openai_args)
            results[ind]["result"] = result
            if args.mode != "tag":
                results[ind]["parsed_result"] = better_json_loads(results[ind]["result"])
            else:
                results[ind]["parsed_result"] = "N/A"
        except Exception as e:
            print(e)
        
        # print("Done!") 
        if ind % args.save_interval == 0 or ind == len(results)-1:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2) 
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def shorten(text, K=-1):
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)"
    return text

def pairwise_eval(args):
    with open(args.first_file, 'r') as f:
        data_1 = json.load(f) 
    with open(args.second_file, 'r') as f:
        data_2 = json.load(f) 
    
    L = min(len(data_1), len(data_2))
    if args.end_idx < 0:
        args.end_idx = L
    print(f"# examples in A: {len(data_1)}; # examples in B: {len(data_2)}; We take {args.end_idx-args.start_idx} for evaluation.")
    data_1 = data_1[:L]
    data_2 = data_2[:L]
    
    results = []
    for itemA, itemB in zip(data_1, data_2):
        assert itemA["instruction"] == itemB["instruction"]
        instruction = itemB["instruction"]
        if random.random() < 0.5:
            itemB["output"], itemA["output"] = itemA["output"], itemB["output"]
            itemB["generator"], itemA["generator"] = itemA["generator"], itemB["generator"]
            
        A, B = itemA["output"], itemB["output"] 
        A, B = shorten(A, args.max_words_to_eval), shorten(B, args.max_words_to_eval)
        if args.mode == "pairwise_multi":
            if args.subtask == 'nc':
                prompt = Template(MULTI_PAIRWISE_TEMPLATE_nc).substitute(
                instruction = instruction, 
                candidateA = A,
                candidateB = B, 
            )
        else:
            prompt = Template(PAIRWISE_TEMPLATE).substitute(
                instruction = instruction, 
                candidateA = A,
                candidateB = B, 
            )
        d = {}
        d["id"] = itemA.get("id", len(results))
        d["input"] = instruction
        d["output_A"], d["output_B"] = itemA["output"], itemB["output"]
        d["generator_A"], d["generator_B"] = itemA["generator"], itemB["generator"]
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        d["prompt"] = prompt
        d["result"] = "N/A"
        results.append(d)
    return results
    
    
def score_eval(args):
    results = []
    with open(args.first_file, 'r') as f:
        candidates = json.load(f) 
    if "+ref" in args.mode:
        with open(args.reference_file, 'r') as f:
            references = json.load(f) 
    else:
        references = [None] * len(candidates)
        
    L = min(len(candidates), len(references))
    if args.end_idx < 0:
        args.end_idx = L
    print(f"# examples in candidates: {len(candidates)}; # examples in references: {len(references)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[:L]
    references = references[:L]
    
    results = []
    for itemA, itemB in zip(candidates, references):
        instruction = itemA["instruction"] 
        if args.mode == "score_multi":
            A = itemA["output"]
            A = shorten(A)
            if args.subtask == "planning":
                prompt = Template(MULTI_SCORE_TEMPLATE_planning).substitute(
                    instruction = instruction, 
                    candidate = A
                )
            elif args.subtask == "conditional":
                prompt = Template(MULTI_SCORE_TEMPLATE_conditional_planning).substitute(
                    instruction = instruction, 
                    candidate = A
                )
            elif args.subtask == "counterfactual":
                prompt = Template(MULTI_SCORE_TEMPLATE_counterfactual_planning).substitute(
                    instruction = instruction, 
                    candidate = A
                )
            elif args.subtask == "nc":
                prompt = Template(MULTI_SCORE_TEMPLATE_nc).substitute(
                    instruction = instruction, 
                    candidate = A
                )
            elif args.subtask == "dimSum":
                prompt = Template(MULTI_SCORE_TEMPLATE_dimSum).substitute(
                    instruction = instruction, 
                    candidate = A
                )
            elif args.subtask == "dimPara":
                prompt = Template(MULTI_SCORE_TEMPLATE_dimPara).substitute(
                    instruction = instruction, 
                    candidate = A
                )
            elif args.subtask == "soda":
                prompt = Template(MULTI_SCORE_TEMPLATE_Soda).substitute(
                    instruction = instruction, 
                    candidate = A
                )
            else:
                print("subtask error....")

        elif args.mode == "score_safety":
            A = itemA["output"]
            A = shorten(A)
            # prompt = Template(SAFETY_SCORE_TEMPLATE).substitute(
            #     instruction = instruction, 
            #     candidate = A
            # )
            prompt = Template(ASR_TEMPLATE).substitute(
                    instruction = instruction, 
                    candidate = A
                )
        else:
            prompt = "N/A" # for reward-based eval 
        d = {}
        d["id"] = itemA.get("id", len(results))
        d["input"] = instruction
        d["output_cand"] = itemA["output"]
        d["generator_cand"] = itemA["generator"]  
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        d["prompt"] = prompt
        d["result"] = "N/A" 
        results.append(d)
    return results 
 

def tag_eval(args):
    results = []
    with open(args.first_file, 'r') as f:
        candidates = json.load(f)  
    references = [None] * len(candidates) 
    L = min(len(candidates), len(references))
    if args.end_idx < 0:
        args.end_idx = L
    print(f"# examples in candidates: {len(candidates)}; # examples in references: {len(references)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[:L]
    references = references[:L]
    
    results = []
    for itemA, itemB in zip(candidates, references):
        instruction = itemA["instruction"] 
        if args.mode == "tag":
            A = itemA["output"]
            A = shorten(A)
            prompt = Template(TAG_DATA_TEMPLATE_V2).substitute(
                instruction = instruction, 
                # candidate = A
            ) 
        d = {}
        d["id"] = itemA.get("id", len(results))
        d["input"] = instruction
        d["output_cand"] = itemA["output"]
        d["generator_cand"] = itemA["generator"]  
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        d["prompt"] = prompt
        d["result"] = "N/A" 
        results.append(d)
    return results 
 


def rm_eval(results, args):
    from just_eval.reward_model import LlamaRewardModel, LlamaTokenizer
    import torch
    results = results[args.start_idx:args.end_idx] # for debug
    print("Loading Reward Model...")
    tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    model = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b", device_map="auto", torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        print("CUDA is available!")
        model = model.to("cuda:0")
    print("Loading Reward Model... Done! ")
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.output_file} "):
        if item["result"] != "N/A":
            results[ind]["parsed_result"] = json.loads(results[ind]["result"])
            print(f"Skipping {ind} for {args.output_file}")
            continue
        # print(f"\nNot Skipping {ind}") 
     
        item["prompt"] = f"""Human: {item['input']}\n Assistant: {item['output_cand']}"""
        inputs = tokenizer(item["prompt"], return_tensors="pt", max_length=1024).to(model.device)
        reward = model(**inputs).item()
        results[ind]["result"] = {"score": reward} 
        
        # print("Done!") 
        if ind % args.save_interval == 0 or ind == len(results)-1:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2) 
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def main():
    random.seed(42)
    args = get_args()
    
    if args.report_only:
        with open(args.output_file) as f:
            results = json.load(f)
        if args.end_idx > 0:
            results = results[:args.end_idx]
        eval_res = report(results, args.mode, args)
        print(json.dumps(eval_res, indent=2))
        with open(args.output_file.replace(".json",".eval_res.json"), "w") as f:
            json.dump(eval_res, f, indent=2)
            print("Evaluation results saved to:", f.name)
        exit()
    
    if args.mode.startswith("pairwise"):
        results = pairwise_eval(args)
        results = gpt_eval(results, args) 
    elif args.mode.startswith("score"):
        results = score_eval(args)
        results = gpt_eval(results, args) 
    elif args.mode.startswith("reward"):
        results = score_eval(args)
        results = rm_eval(results, args) 
    elif args.mode == "tag":
        results = tag_eval(args)
        results = gpt_eval(results, args)

if __name__ == "__main__": 
    main()
     