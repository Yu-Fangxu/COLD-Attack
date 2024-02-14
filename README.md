# Code for "COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability"

This is the code for the following paper: 

[COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability] ([https://arxiv.org/pdf/2202.11705.pdf](https://arxiv.org/abs/2402.08679)) \
Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

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
