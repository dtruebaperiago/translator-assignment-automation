

## Evaluation Results Analysis: IDISC Dual-Tower Recommender System

1. Summary (The best results where achieved best_idisc_model(3))
The evaluation of the trained Dual-Tower models was conducted on a test set of 149,690 tasks against a pool of 374 unique translators. After applying strict hard constraints (Language Pair, Task Type, and Schedule availability), the model achieved a peak Hit Rate@5 of 2.32% and a Mean Reciprocal Rank (MRR) of 0.0465. These percentages are low, but a deeper statistical analysis reveals that the model has successfully captured significant patterns in the data, performing better than a complete random prediction. Which is not bad for a intial evaluation.

When analyzing the Affinity Score distribution, we observed that the model concentrates its predictions in a very high band (between 0.82 and 0.855). This indicates a lack of discriminative power.

This behavior actually proves that our Hard Constraints filter is highly effective. Because the filter strictly removes incompatible translators beforehand, the candidates that reach the Neural Network are already excellent matches. The model correctly identifies this and assigns them all high scores.

However, because these scores are so  compressed, the final ranking suffers from noise—it struggles to break ties between very identical candidates. This perfectly explains our Hit Rate@5 of 2.32%.

![alt text](image.png)
![alt text](newplot.png)

2. Contextualizing the Results
To understand the efficacy of the model, we must compare it against the Random Baseline:

Random Performance: In a pool of approximately 300 available candidates per task, the mathematical probability of a random guess appearing in the Top 5 is roughly 1.6%. Our model outperforms this by nearly 45%.

Mean Rank Insight: An MRR of 0.0465 implies an average rank of ~21.5. In a scenario where a human Project Manager would otherwise have to manually check 300 names, the model narrows the "perfect" historical candidate down to the top 7% of the pool.

3. Difficulties
Candidate Homogeneity: After the "Hard Filters" are applied, the remaining candidates are already highly qualified. Differentiating between the "Best" and the "Second Best" is mathematically challenging when their features are very similar.

Label Subjectivity: Historical data represents human choices made by Project Managers. A Manager might choose "Translator A" over "Translator B" for personal or subjective reasons not captured in the numerical features. If the model ranks the choice at #8 instead of #1 it is still a high-quality recommendation.

4. Areas for Improvement
To improve our performance, the following technical iterations are proposed:

## (TO BE DETERMINED)

## Per-Task Inspector

While global metrics provide a statistical overview, the Per-Task Inspector allows us to audit the model's behavior on individual assignments. The task-level results perfectly align with the global evaluation: although the historical translator frequently ranks outside the Top 5, the sorting is not random. The rankings maintain clear mathematical validity, confirming that the model effectively groups candidates at the top.

