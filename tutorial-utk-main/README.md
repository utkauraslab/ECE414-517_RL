# 🎓 ECE 414 / 517 — Reinforcement Learning

**University of Tennessee, Knoxville | Fall 2026**

Welcome to the course tutorial repository for **ECE 414 / 517: Reinforcement Learning**.

This repository contains the practical coding tutorials used in the course, including student exercise notebooks, reference solutions, and the supporting files needed to run them in Google Colab or locally.

## 🚀 Interactive Tutorials

You do not need to install anything on your local machine to use these tutorials. The notebooks can run directly in Google Colab.

| 📚 Tutorial Module | 📝 Description | 💻 Student Exercise | 🔒 Reference Solution |
| :----------------- | :------------- | :------------------ | :-------------------- |
| **Basics of RL I** | Gymnasium API, tabular RL, prediction, and control. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkauraslab/ECE414-517_RL/blob/main/RL-basics/RL01_Intro_to_RL.ipynb) | [Open Solution in Colab](https://colab.research.google.com/github/utkauraslab/ECE414-517_RL/blob/main/RL-basics/RL01_Intro_to_RL_solution.ipynb) |
| **Basics of RL II** | Policy gradient algorithms. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkauraslab/ECE414-517_RL/blob/main/policy-gradient/RL02__Policy_gradient_methods.ipynb) | [Open Solution in Colab](https://colab.research.google.com/github/utkauraslab/ECE414-517_RL/blob/main/policy-gradient/RL02__Policy_gradient_methods_solution.ipynb) |

> **Note:** The passcode for accessing the reference solutions will be provided during class.

## 🛠️ Instructions for Students

1. **Open the notebook:** Click the Colab link for the tutorial you want to use.
2. **Save your own copy:** In Colab, select **`File > Save a copy in Drive`** before making changes.
3. **Run the notebook:** Execute the cells in order and complete the marked TODO sections.
4. **Review the results:** Use the generated outputs and visualizations to understand the RL algorithm behavior.

## 💻 Optional Local Setup

If you prefer to work locally, clone this repository and create a Python environment:

```bash
git clone https://github.com/utkauraslab/ECE414-517_RL.git
cd ECE414-517_RL

conda create -n ece414517-rl python=3.11
conda activate ece414517-rl
pip install -r requirements.txt
