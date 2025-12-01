# mlinsightx-agent
Train -> Explain -> Narrate -> Evaluate - Fully Automated with AI Agents 

**MLInsightX — A fully automated, multi-agent ML system that trains models, generates explainability artifacts, produces stakeholder narratives, and self-evaluates model quality using Gemini + Google ADK. It takes a tabular dataset, trains a model, generates explainability artifacts, narrates the results for stakeholders, evaluates the quality of that narrative, and packages everything into reusable outputs. This project started from my actual pain as a data scientist: I kept repeating the same pattern (train → explain → summarize) across projects. MLInsightX is my attempt to automate that loop using agents. Built as the final project for the Kaggle 5-Day AI Agent Intensive Course powered by Google.**

## What It Does
Given a tabular dataset (e.g., the Bank Marketing dataset), MLInsightX:
1. **Trains a classifier** (LightGBM in a scikit-learn `Pipeline`).
2. **Computes explainability artifacts**:
  - SHAP values (TreeExplainer on transformed features)
  - permutation importance
  - PDP plots for key features.
3. **Builds a global explanation** (what drives predictions, which features matter most).
4. **Uses Gemini to narrate the insights** in stakeholder-friendly language.
5. **Uses Gemini again as a Judge Agent** to score the narrative for:
  - clarity, correctness, completeness, actionability.
6. **Packages the outputs** into:
  - `mlinsightx_report.json`
  - `mlinsightx_report.pdf`
  - a folder of PDP plots.
On top of that, the project wires these pieces into a **multi-agent system** using Google ADK, so the whole flow feels like talking to one orchestrator agent.
---
## Architecture
**Core flow:**
1. **Trainer Agent (Python)**  
  - Trains the LightGBM pipeline.  
  - Computes metrics: accuracy, F1, ROC AUC.  
  - Stores results in a shared `artifacts` dict.
2. **Explainer Agent (Python + Tools)**  
  - `compute_shap_values`  
  - `compute_permutation_importance`  
  - `compute_pdp` (safe on transformed features, skips constant ones).  
  - Produces a compact “global explanation” summary.
3. **Narrative Agent (Gemini via ADK)**  
  - Takes metrics + global explanation.  
  - Generates stakeholder-facing text.
4. **Judge Agent (Gemini-as-judge)**  
  - Scores the narrative and returns a small JSON verdict.
5. **Reporter Agent**  
  - Writes everything to JSON and saves PDP plots.
6. **Orchestrator Agent (LlmAgent with tools)**  
  - Single entry point for questions like:
> “Give me a full overview of the model and its insights.”
---
## 🧩 ADK Features Used
- **Multi-agent pattern** (Trainer, Explainer, Narrative, Judge, Reporter, Orchestrator).  
- **Tools**:
 - Custom function tools for metrics, SHAP, perm, and PDP summaries.
- **Sessions & Memory**:
 - `InMemoryRunner` with in-memory session/memory services.
 - Shared `artifacts` dict as a simple memory bank in the notebook.
- **Context Engineering**:
 - Compact prompts for Gemini.
- **Agent Evaluation**:
 - LLM-as-judge scoring of the narrative.
---
## Repo Structure
```bash
MLInsightX/
├── README.md
├── MLInsightX_Kaggle_Notebook.ipynb
├── MLInsightX Agent Architecture.pdf
├── thumbnail.png
├── mlinsightx_report.json
├── mlinsightx_report.pdf

