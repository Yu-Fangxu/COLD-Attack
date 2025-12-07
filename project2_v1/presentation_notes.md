# Presentation Notes
## COLD-Attack - CS553 Project 2

**Time:** 15 minutes
**Team:** Josh, Jordan, Ian, Chao

---

## Slide 1: Title

**COLD-Attack: Controllable Adversarial Attacks on LLMs**

Team 2: Joshua Howard, Ian Gower, Jordan Spencer, Chao Jung Wu

CS553 Neural Networks | Fall 2024 | SDSU

---

## Slide 2: Intro & Motivation

**Main points:**
- LLMs are trained to refuse harmful requests (RLHF)
- But researchers found ways to bypass this
- Red-teaming = finding vulnerabilities to build better defenses
- Old attacks like GCG make gibberish text - easy to detect
- COLD-Attack makes natural sounding text

**Speaker:** Chao (~2 min)

---

## Slide 3: Problem & Related Work

**Problem:** How to make attacks that are:
1. Effective (high success rate)
2. Stealthy (natural text)
3. Controllable (can specify style)

**Comparison:**

| Method | Effective | Stealthy | Controllable |
|--------|-----------|----------|--------------|
| GCG | yes | no | no |
| AutoDAN | yes | kinda | no |
| COLD-Attack | yes | yes | yes |

**Speaker:** Ian (~2.5 min)

---

## Slide 4: Approach

**Energy function:**
```
E = attack_loss + fluency_loss + rejection_loss
```

**Langevin dynamics:**
```
y_new = y_old - stepsize * gradient + noise
```

- Start with random text
- Iterate 2000 times
- Noise helps escape local minima

[Use Figure 3 from paper here]

**Speaker:** Josh (~3 min)

---

## Slide 5: Experiments

**Setup:**
- Model: Vicuna-7B-v1.5
- Dataset: AdvBench (100 prompts)
- Hardware: Colab A100
- Iterations: 1000

**Metrics:**
- ASR = attack success rate
- PPL = perplexity (fluency)

**Our differences:**
- Used SPSA instead of backprop
- Fewer iterations

**Speaker:** Jordan (~2.5 min)

---

## Slide 6: Results

[Put graphs here]
- ASR comparison bar chart
- Loss curves
- PPL distribution

**Key findings:**
- Got X% ASR (paper got ~90%)
- Gap probably from SPSA vs real gradients
- GPT-4 selection helps a lot

**Speaker:** Jordan/Ian (~3 min)

---

## Slide 7: Conclusion

**Takeaways:**
1. COLD-Attack generates stealthy jailbreaks
2. Energy-based framework works well
3. Implementation details matter a lot

**Limitations:**
- SPSA adds noise
- Need good GPU

**Team contributions:**
- Josh: Implementation, GPU stuff
- Ian: Evaluation, metrics
- Jordan: Analysis, graphs
- Chao: Presentation, writing

**Speaker:** Everyone (~1.5 min)

---

## Timing

| Slide | Who | Time |
|-------|-----|------|
| 1 | Chao | 30s |
| 2 | Chao | 2 min |
| 3 | Ian | 2.5 min |
| 4 | Josh | 3 min |
| 5 | Jordan | 2.5 min |
| 6 | Jordan/Ian | 3 min |
| 7 | All | 1.5 min |
| **Total** | | **15 min** |

---

## Questions to prep for

- "Why not use the original code?" - We wanted to understand it better + had GPU issues
- "How does this compare to prompt injection?" - Different technique, COLD uses optimization
- "Can defenses detect these?" - Good question, could use perplexity filters or Llama Guard
