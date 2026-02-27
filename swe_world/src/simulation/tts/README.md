
# Test-Time Scaling (TTS) with SWR

This directory provides a Docker-free Test-Time Scaling (TTS) pipeline using the SWE-World Reward Model (SWR).

TTS workflow:

1. Sample K trajectories (collect mode)
2. Extract reward contexts
3. Compute simulated rewards
4. Select best candidate and compute Pass@K

---

## 1️⃣ Sample K Trajectories

Run the SWE agent in `collect` mode and sample K times per instance:

```bash
bash ../../../scripts/run_mode_collect.sh
````

Make sure `execution_mode=collect`.

---

## 2️⃣ Collect Reward Context

Extract reward contexts from trajectories:

```bash
python 0_collect_reward_context.py
```

---

## 3️⃣ Compute Simulated Reward

Multi-score version (recommended):

```bash
python 1_get_swr_multi_reward.py
```

Single-score version:

```bash
python 1_get_swr_single_reward.py
```

Multi-score mode:

* Runs SWR multiple times
* Filters invalid simulations
* Computes `simulated_reward_avg`

---

## 4️⃣ Compute Final TTS Result (Pass@K)

Select the candidate with the highest `simulated_reward_avg` per `instance_id`:

```bash
python 2_get_tts_score.py
```

Outputs:

* Final Pass@K (reward=1 ratio)
* Average effective K
* Selected best candidates (JSONL)

---

Conceptually:

Issue → {Trajectory₁ ... Trajectoryₖ}
→ SWR scoring
→ Select max avg score
→ Final reward
