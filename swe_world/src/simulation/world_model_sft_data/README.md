
## SFT World Models

This section describes how to generate Supervised Fine-Tuning (SFT) data for the SWE-World **Reward Model (SWR)** and **Transition Model (SWT)**.

### 1. Generate Initial Analysis (Optional)

First, generate the `initial_analysis` field for each data sample:

```bash
bash swe_world/src/simulation/world_model_sft_data/0_run_get_initial_analyse.sh
```

---

### 2. Collect SFT Data

Modify the script to specify whether you are generating data for the **reward model** or **transition model**, then run:

```
swe_world/src/simulation/world_model_sft_data/1_collect_world_model_sft_data.py
```

After data generation, you can proceed with SFT training.

---

## CoT Version (Optional)

If you want to generate Chain-of-Thought (CoT) enhanced SFT data:

### Step 1: Collect Data

* Reward model:

```
swe_world/src/simulation/world_model_sft_data/1_collect_world_model_sft_cot_data_reward.py
```

* Transition model:

```
swe_world/src/simulation/world_model_sft_data/1_collect_world_model_sft_cot_data_transition.py
```

### Step 2: Generate Simulated CoT

```bash
swe_world/src/simulation/world_model_sft_data/2_run_get_sim_cot.sh
```

### Step 3: (Optional) Filter Low-Quality CoT

Use LLM-as-a-Judge to remove low-quality reasoning traces:

```bash
swe_world/src/simulation/world_model_sft_data/3_run_llm_as_judge_for_sim_cot.sh
```

---

## 3. Run SFT

Once the dataset is prepared, you can train the model using frameworks such as **OpenRLHF**.
