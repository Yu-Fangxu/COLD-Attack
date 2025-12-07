# Project Notes - COLD-Attack Replication
## CS553 Neural Networks - Team 2

**Team:** Josh, Jordan, Ian, Chao Jung Wu
**Due:** Tuesday, December 10, 2024

---

## Current Status

| What | Current | Target | Notes |
|------|---------|--------|-------|
| ASR | 66.7% | ~90% | lower than paper |
| Instances | 3 | 100 | need to run more |
| Iterations | 500 | 2000 | could bump up |
| Samples/prompt | 1 | 8 | missing the selection step |

### Why our results are different

1. **Gradient method** - Paper uses real backprop, we use SPSA (gradient estimation). Adds noise to the optimization.

2. **Fewer iterations** - Running 500 vs 2000 means less time to converge

3. **No multi-sample selection** - The paper generates 8 candidates per prompt and uses GPT-4 to pick the best one. We just take whatever comes out.

4. **Simpler decoding** - Paper has fancier top-k stuff, we simplified it

---

## Todo List

### Need to do
- [ ] Get multi-sample generation working
- [ ] Add GPT-4 selection step
- [ ] Run more iterations (1000-2000)
- [ ] Fix PPL calculation (shows NaN)
- [ ] Run 100 instances
- [ ] Make graphs

### Presentation stuff
- [ ] Slides 1-4: intro, problem, related work, approach
- [ ] Slides 5-6: results and what we learned
- [ ] Slide 7: conclusion and contributions

---

## How the attack works

Basic flow:
```
Harmful prompt -> Langevin optimization -> Adversarial suffix -> Full prompt -> Model responds
```

Energy function:
```
E = attack_loss + fluency_loss + rejection_loss
```

- attack_loss = does model comply?
- fluency_loss = is text natural?
- rejection_loss = avoid "sorry", "cannot", etc.

---

## Schedule

- **Saturday:** get multi-sample working, test on few instances
- **Sunday:** run overnight, start graphs
- **Monday:** finish graphs, practice presentation
- **Tuesday:** present

---

## If stuff breaks

- If selection doesn't help: focus on explaining the SPSA vs backprop difference
- If can't run 100: 50 should be fine
- If code crashes: use what we have, explain in presentation

---

## Resources

- Paper: in project files
- Original repo: https://github.com/Yu-Fangxu/COLD-Attack
- GCG paper: https://arxiv.org/abs/2307.15043
