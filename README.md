# COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability

We study the **controllable** jailbreaks on large language models (LLMs). Specifically, we focus on how to enforce control on LLM attacks. In this work, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the [Energy-based Constrained Decoding with Langevin Dynamics (COLD)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3e25d1aff47964c8409fd5c8dc0438d7-Abstract-Conference.html), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios including:
1. Fluent suffix attacks (standard attack setting which append the adversarial prompt to the original malicious user query).
2. Paraphrase attack with and without sentiment steering (revising a user query adversarially with minimal paraphrasing).
3. Attack with left-right-coherence (inserting stealthy attacks in context with left-right-coherence).

More details can be found in our paper:
Xingang Guo*, Fangxu Yu*, Huan Zhang, Lianhui Qin, Bin Hu, "[COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability](https://arxiv.org/abs/2402.08679)" (* Equal contribution)

## COLD-Attack

![plot](./imgs/COLD_attack_diagram.png)

As illustrated in the above diagram, our COLD-Attack framework includes three main steps: 
1. **Energy function formulation**: specify energy functions properly to capture the attack constraints such as fluency, stealthiness, sentiment, and left-right-coherence.
2. **Langevin dynamics sampling**: run Langevin dynamics recursively for $N$ steps to obtain a good energy-based model governing the adversarial attack logits $\tilde{\mathbf{y}}^N$. 
3. **Decoding process**: leverage an LLM-guided decoding process to covert the continuous logit $\tilde{\mathbf{y}}^N$ into discrete text attacks $\mathbf{y}$. 

## Jailbreaking Performance
We evaluate the performance of COLD-Attack on four popular white-box LLMs: Vicuna-7b-v1.5 (Vicuna), Llama-2-7b-Chat-hf (Llama2), Guanaco-7b (Guanaco), and Mistral-7b-Instruct-v0.2. In addition, we use the following three main evaluation metrics:
1. Attack Successful Rate (**ASR**): the percentage of instructions that elicit corresponding harmful outputs using sub-string matching.
2. GPT-4 based ASR (**ASR-G**): We develop a prompt template and utilize GPT-4 to assess whether a response accurately fulfills the malicious instruction. Based on our observations, ASR-G has shown higher correlation with human annotations, providing a more reliable measure of attack effectiveness.
3. Perplexity (**PPL**): We use PPL to evaluate the fluency of the generated prompts and use Vicuna-7b to do the PPL calculation.

### Fluent suffix attack

| Models   | ASR ↑  | ASR-G ↑ | PPL ↓ |
|----------|--------|---------|-------|
| Vicuna   | 100.0  | 86.00   | 32.96 |
| Guanaco  | 96.00  | 84.00   | 30.55 |
| Mistral  | 92.00  | 90.00   | 26.24 |
| Llama2   | 92.00  | 66.00   | 24.83 |


### Paraphrase attack 

| Models   | ASR ↑  | ASR-G ↑ | PPL ↓ | BLEU ↑ | ROUGE ↑ | BERTScore ↑ |
|----------|--------|---------|-------|--------|---------|--------------|
| Vicuna   | 96.00  | 80.00   | 31.11 | 0.52   | 0.57    | 0.72         |
| Guanaco  | 98.00  | 78.00   | 29.23 | 0.47   | 0.55    | 0.74         |
| Mistral  | 98.00  | 90.00   | 37.21 | 0.41   | 0.55    | 0.72         |
| Llama2   | 86.00  | 74.00   | 39.26 | 0.60   | 0.54    | 0.71         |

### Left-right-coherence control

| Models                  | ASR ↑ | ASR-G ↑ | Succ ↑ | PPL ↓ |
|-------------------------|-------|---------|--------|-------|
| **Sentiment Constraint**|       |         |        |       |
| Vicuna                  | 90.00 | 96.00   | 84.00  | 66.48 |
| Guanaco                 | 96.00 | 94.00   | 82.00  | 74.05 |
| Mistral                 | 92.00 | 96.00   | 92.00  | 67.61 |
| Llama2                  | 80.00 | 88.00   | 64.00  | 59.53 |
| **Lexical Constraint**  |       |         |        |       |
| Vicuna                  | 92.00 | 100.00  | 82.00  | 76.69 |
| Guanaco                 | 92.00 | 96.00   | 82.00  | 99.03 |
| Mistral                 | 94.00 | 84.00   | 92.00  | 96.06 |
| Llama2                  | 88.00 | 86.00   | 68.00  | 68.23 |
| **Format Constraint**   |       |         |        |       |
| Vicuna                  | 92.00 | 94.00   | 88.00  | 67.63 |
| Guanaco                 | 92.00 | 94.00   | 72.00  | 72.97 |
| Mistral                 | 94.00 | 86.00   | 84.00  | 44.56 |
| Llama2                  | 80.00 | 86.00   | 72.00  | 57.70 |
| **Style Constraint**    |       |         |        |       |
| Vicuna                  | 94.00 | 96.00   | 80.00  | 81.54 |
| Guanaco                 | 94.00 | 92.00   | 70.00  | 75.25 |
| Mistral                 | 92.00 | 90.00   | 86.00  | 54.50 |
| Llama2                  | 80.00 | 80.00   | 68.00  | 58.93 |

Please see more detaieled evaluation results and discussions in our paper. 

## Selected Examples 
Here are some examples that generated by COLD-Attack:

![plot](./imgs/selected_samples_2.png)


## Code
**1) Download this GitHub**
```
git clone https://github.com/Yu-Fangxu/COLD-Attack.git
```

**2) Setup Environment**
```
pip install -r requirements.txt
```

**3) Run Command for COLD-Attack**

* Fluent suffix attack
```
bash attack.sh "suffix"
```

* Paraphrase attack
```
bash attack.sh "paraphrase"
```

* Left-right-coherence control
```
bash attack.sh "control"
```

<br> **If you find our repository helpful to your research, please consider citing:** <br>
```
@article{guo2024cold,
  title={COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability},
  author={Guo, Xingang and Yu, Fangxu and Zhang, Huan and Qin, Lianhui and Hu, Bin},
  journal={arXiv preprint arXiv:2402.08679},
  year={2024}
}
```
```
@article{qin2022cold,
  title={Cold decoding: Energy-based constrained text generation with langevin dynamics},
  author={Qin, Lianhui and Welleck, Sean and Khashabi, Daniel and Choi, Yejin},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9538--9551},
  year={2022}
}
```
