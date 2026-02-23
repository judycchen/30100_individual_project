# Predicting Depression from Worry-Stress Profiles
**30100 ML for Computational Social Science — Individual Project**
Author: Judy Chen

## Project Overview

This project examines whether worry-stress profiles can predict clinical depression (PHQ-9 >= 10) in Chinese university students (n = 24,292), and whether distinct worry-stress phenotypes emerge from the data. Predictors are item-level responses from three scales: GAD-7 (anxiety/worry), PSS-14 (perceived stress), and ISI (insomnia severity).

## Repository Structure

| File / Folder | Description |
|---|---|
| `code.ipynb` | Main notebook - EDA, supervised learning, unsupervised learning |
| `figures/` | All output plots saved during notebook execution |
| `scale_items.csv` | Item text and response labels for all 28 predictor variables |
| `next_submission_plan.md` | Roadmap for Section IV (Result Analysis) and Section V (Conclusion), to be completed in the final submission |

## Notebook Structure

| Section | Content |
|---|---|
| **I - EDA** | Data loading & merging, quality checks, descriptive statistics, target distribution, feature encoding, collinearity analysis |
| **II - Supervised Learning** | Logistic Regression (LR-1–7) and Decision Tree (DT-1–5) with progressively more sophisticated imbalance-handling strategies; model comparison; reflection on best model (LR-7) |
| **III - Unsupervised Learning** | PCA (dimensionality reduction + interpretation), K-Means clustering (elbow method, k=4 phenotypes), internal and external cluster evaluation |

## Current Submission Status

Sections I-III are complete. Section IV (Result Analysis) and Section V (Summary & Conclusion) are planned for the final submission — see `next_submission_plan.md` for the roadmap.
