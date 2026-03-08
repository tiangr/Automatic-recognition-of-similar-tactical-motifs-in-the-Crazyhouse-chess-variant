# Version 1
PGN sample size: 500 games
Mined tactical events: 1000
Augmented positions: 1000
Filtered tactical positions: 987 (Dropped the ones that don't include figure drops)
Corpus size: 987

BM25 field tested: text_static, text_dynamic, text_all
top-k: 10
relevance definition: same (bestmove_before, dyn:first:action), different game -> A retrieved position is correct if it has the same recommended tactical move and the same kind of first tactical idea, but it must come from another game.



## Evaluation results
text_dynamic: Recall@10 = 0.433, MRR@10 = 0.282

text_all: Recall@10 = 0.441, MRR@10 = 0.268

text_static: Recall@10 = 0.102, MRR@10 = 0.036

dynamic representation clearly outperforms static-only retrieval
combined representation slightly improves recall
retrieved examples often share the same drop idea and similar PV structure
synthetic nearby-ply experiment behaved as expected but is only diagnostic

## top-k testing
Number of usable queries: **127**

| Field        | Recall@5 | Recall@10 | Recall@20 | MRR@5 | MRR@10 | MRR@20 |
|--------------|---------:|----------:|----------:|------:|-------:|-------:|
| text_dynamic | 0.370    | 0.433     | 0.535     | 0.273 | 0.282  | 0.288  |
| text_all     | 0.362    | 0.441     | 0.567     | 0.257 | 0.268  | 0.276  |
| text_static  | 0.063    | 0.102     | 0.150     | 0.031 | 0.036  | 0.039  |

