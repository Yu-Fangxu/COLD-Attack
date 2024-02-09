# Code for "COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability"

This is the code for the following paper: 

[COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability] (https://arxiv.org/pdf/2202.11705.pdf) \
Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu


**1) Setup Environment**
```
pip install -r requirements.txt
```

**2) Download this GitHub**
```
git clone https://github.com/Yu-Fangxu/COLD-Attack.git
```

**3) Run Command for COLD-Attack**

* Suffix Attack
```
sh commongen.sh
```

* Paraphrasing Attack 
```
sh abductive.sh
```

* Attack with Control
```
sh counterfactual.sh
```
