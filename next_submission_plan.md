# Next Submission Plan

## Section IV: Result Analysis
I just haven't written about this systematically, so I'll be focusing on this (and the presentation, of course) in the next submission

**4.1 Interpret model performance**
- DT: discuss `dt_structure.png` — which feature splits first, what the top nodes reveal about which domain is most diagnostic
- LR: redo coefficient plot on LR-3/LR-7, label with item text from `scale_items.csv`, check if signs make clinical sense (PSS positive items should be negative coefficients)

**4.2 Error analysis**
- Profile FN (missed depressed students) and FP (flagged non-depressed) for LR-7
- Compare where LR-7 and DT-5 disagree — connect to each model's known weaknesses
- Suggest improvements: ensemble methods, probability calibration, PSS reverse-scoring


## Section V: Summary & Conclusion

- Supervised: LR-7 ROC-AUC = 0.961 → worry-stress profiles predict depression
- Unsupervised: Cluster 3 has 9× higher depression rate, discovered without PHQ-9 labels → phenotypes are real
- Key finding: stress alone (Cluster 0, 0.8% depression) does not predict depression — co-occurrence across domains does
- Limitations: cross-sectional, self-report, single site, binary PHQ-9 cutoff
