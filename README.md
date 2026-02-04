<h1 align="center">SWE-World: Building Software Engineering Agents in Docker-Free Environments</h1>

<div align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code_License-MIT-blue" alt="license"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Model_License-MIT-blue" alt="license"></a>
  <!-- TODO: replace with your paper link -->
  <a href="https://arxiv.org/pdf/2602.03419" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv"></a>
  <!-- TODO: replace with your HF collection/model link if any -->
  <a href="https://huggingface.co/" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
</div>

<h5 align="center">If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

---

## ✨ News
+ [4 Feb 2026] ⚡️⚡️ [**SWE-World**](https://arxiv.org/pdf/2602.03419): We introduce SWE-World, a fully Docker-free framework that replaces physical execution environments with learned surrogates. It lifts Qwen2.5-Coder-32B from 6.2% to 55.0% on SWE-bench Verified via fully Docker-free agentic SFT and RL, and further attains 68.2% through test-time scaling (TTS@8).

+ [4 Feb 2026] ⚡️⚡️ [**SWE-Master**](https://arxiv.org/pdf/2602.03411): We introduce SWE-Master, a fully reproducible post-training framework for Qwen2.5-Coder-32B that integrates agentic SFT and RL to achieve 61.4% (Pass@1) and 70.8% (TTS@8) resolve rates on SWE-bench Verified. Meanwhile, the framework incorporates IDE-level capabilities during inference via LSP-driven tool.
---

## 💡 Overview
SWE agents typically rely on containerized execution environments (e.g., Docker) to obtain step-level execution feedback and final unit-test results. While effective, Docker-based pipelines are expensive and brittle at scale. **SWE-World** replaces physical execution with a *learned surrogate world*: a sandbox lightweight handles file-system edits, an LLM-based **Transition Model (SWT)** simulates step-level execution feedback for commands that would otherwise require Docker, and an LLM-based **Reward Model (SWR)** acts as a virtual test runner that produces structured test feedback and a binary success signal. This enables **fully Docker-free** data generation, supervised fine-tuning (SFT), reinforcement learning (RL), and test-time scaling (TTS) for repository-level issue resolution.

<p align="center">
  <img src="assets/fig1_overview.png" width="95%" alt="Overview of SWE-World">
</p>

---

## ✨ Key Insights
- Docker-Free SWE Environment: We propose SWE-World, a Docker-free framework that replaces physical execution environments with a learned surrogate for training and evaluating software engineering agents.

- Effective Agent Training without Execution: We show that LLM-based environment feedback can successfully support SFT and RL, achieving performance comparable to or better than training with real execution.

- Scalable Use of Open-Source SWE Data: By eliminating the requirement for buildable environments, SWE-World enables substantially broader utilization of real-world GitHub data for training software engineering agents. 

---

## ✨ Method
SWE-World has two parts: **(1) Build SWE-World** (train world models), and **(2) Train SWE Agents with SWE-World** (use SWE-World to generate trajectories and optimize agents).

### 1) Build SWE-World
- **Sandbox**: executes Navigation & Editing actions (e.g., `ls`, `cat`, `grep`, `str_replace`).
- **SWT (SWE-World Transition Model):** predicts step-level execution feedback (e.g., stdout/stderr/exit status) for code execution commands.
- **SWR (SWE-World Reward Model):** replaces containerized unit-test runs at the end of a trajectory; it generates a structured test report and outputs a binary reward.


### 2) Train SWE Agents with SWE-World (Docker-Free SFT & RL)
- **Data Preparation:** we construct a unified instance pool from (i) open-source SWE datasets and (ii) a newly curated **SWE-World Dataset** (16.6K tasks across 3,763 repositories).
- **Docker-Free SFT:** we roll out powerful code agents inside SWE-World to generate trajectories, then apply filtering (rule-based + SWR-based) and perform agentic SFT.
- **Docker-Free RL:** starting from the SFT checkpoint, we run RL where SWT provides step-level feedback and SWR provides rewards.
- **Docker-Free Test-Time Scaling (TTS):** for each issue instance, sample multiple candidate trajectories, use SWR to score them, and submit the best candidate.

---

## 📄 Overall Performance

### Main results on SWE-bench Verified (Resolve Rate %)
Our fully Docker-free pipeline substantially improves strong open-source backbones, and SWR enables effective Docker-free test-time scaling:

<p align="center">
  <img src="assets/fig2_performance.png" width="95%" alt="Overall Performance">
</p>

> Notes: Resolve Rate is measured via the official SWE-bench Verified Docker evaluation harness; TTS@8 means selecting the best patch among 8 sampled candidates using SWR.


### Performance of SWT and SWR

- **SWT:** SWT-72B best closes the sim-to-real gap, supporting **60.2%** resolve rate for Minmax M2.1, higher than GLM-4.7 (**59.4%**) and Minmax-M2.1 (**56.2%**).
- **SWR:** SWR-32B is strong and precise (Acc **0.754**, Prec **0.779**). Scaling to SWR-72B further improves (Acc **0.770**, Prec **0.780**), yielding the best comprehensive reward simulation.

<p align="center">
  <img src="assets/fig2_performance_swt_swr.png" width="95%" alt="Performance of SWT and SWR">
</p>

---

## 🔎 Analysis

### 1) Impact of Chain-of-Thought (CoT)
We train and compare the performance of SWT/SWR with and without CoT in their outputs. CoT provides **asymmetric benefits**: it yields only marginal gains for SWT, but **substantially improves** SWR’s reward prediction quality (Accuracy/Precision/Recall/F1), making the reward signal more reliable.

<p align="center">
  <img src="assets/fig3_impact_cot.png" width="90%" alt="RL training dynamics">
</p>

### 2) RL Training Dynamics (Stability vs. Reward Hacking)
With CoT-enhanced SWR, Docker-free RL learns stably. Without CoT, training can collapse due to reward hacking (short invalid solutions mistakenly rewarded).

<p align="center">
  <img src="assets/fig4_rl.png" width="90%" alt="RL Training Dynamics">
</p>

### 3) Test-Time Scaling (TTS)
SWR provides a strong ranking signal: performance increases monotonically with K and reaches 68.2% at TTS@8, outperforming prior verifiers under the same setting.

<p align="center">
  <img src="assets/fig5_tts.png" width="90%" alt="Test-Time Scaling">
</p>

### 4) Qualitative Fidelity of Simulation
We provide side-by-side comparisons between real Docker outputs and SWT/SWR simulated outputs for both step-level feedback and final test reports.

<p align="center">
  <img src="assets/fig6_case.png" width="90%" alt="Qualitative Fidelity of Simulation">
</p>

---

## 🏃 Quick Start
The training and inference frameworks are currently undergoing internal corporate review to ensure compliance with open-source policies. We are committed to releasing them as soon as the process is finalized.

---

## 📄 Citation
If you find SWE-World useful, please cite our paper:

```bibtex
@misc{sun2026sweworldbuildingsoftwareengineering,
      title={SWE-World: Building Software Engineering Agents in Docker-Free Environments}, 
      author={Shuang Sun and Huatong Song and Lisheng Huang and Jinhao Jiang and Ran Le and Zhihao Lv and Zongchao Chen and Yiwen Hu and Wenyang Luo and Wayne Xin Zhao and Yang Song and Hongteng Xu and Tao Zhang and Ji-Rong Wen},
      year={2026},
      eprint={2602.03419},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.03419}, 
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [sunshuang@ruc.edu.cn](sunshuang@ruc.edu.cn).