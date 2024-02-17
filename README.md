# COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability

We study the **controllable** jailbreaks on large language models (LLMs). Specifically, we focus on how to enforce control on LLM attacks. In this work, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios including:
1. Fluent suffix attacks (standard attack setting which append the adversarial prompt to the original malicious user query).
2. Paraphrase attack with and without sentiment steering (revising a user query adversarially with minimal paraphrasing).
3. Attack with left-right-coherence (inserting stealthy attacks in context with left-right-coherence).

More details can be found in our paper:
Xingang Guo*, Fangxu Yu*, Huan Zhang, Lianhui Qin, Bin Hu, "[COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability](https://arxiv.org/abs/2402.08679)" (* Equal contribution)[https://arxiv.org/pdf/2202.11705.pdf](https://arxiv.org/abs/2402.08679) 

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


---

<br> **If you find our repository helpful to your research, please consider citing:** <br>
```
@article{guo2024cold,
  title={COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability},
  author={Guo, Xingang and Yu, Fangxu and Zhang, Huan and Qin, Lianhui and Hu, Bin},
  journal={arXiv preprint arXiv:2402.08679},
  year={2024}
}
```
