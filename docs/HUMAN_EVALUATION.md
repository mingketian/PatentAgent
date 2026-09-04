# Human Evaluation of Grounded Answers

Retrieval metrics measure ranking. They say nothing about whether the *answer* a user reads is
trustworthy. This document describes the manual study that measures that directly.

Raw data lives in [`eval/`](eval); derived tables in [`../results/tables/`](../results/tables).

---

## 1. Design

| | |
|---|---|
| Prompts | 20, from [`eval/product_qa_queries.json`](eval/product_qa_queries.json) |
| Systems | 4 |
| Answers labelled | 80 (20 × 4) |
| Aspects per answer | 4 (groundedness, helpfulness, hallucination, context reuse) |
| Annotation | By the project team, after all four systems had answered |

### Systems compared

| ID | System | What it is |
|---|---|---|
| `our_agent` | **Our Agent** | The full pipeline: recall → learned rerank → planner → evidence → grounded answer → verification |
| `rag_only` | **RAG only** | The in-repo baseline: dense retrieval from the same index, top-3 patents, no synthesis |
| `chatgpt_auto` | **ChatGPT Auto** | A general assistant answering the same prompt with its own tools |
| `gemini_fast` | **Gemini Fast** | A second general assistant, same conditions |

The two general assistants are the fair comparison for *usefulness*; `RAG only` is the fair
comparison for *what our own retrieval stack gives you without the agent on top*.

### Prompt categories

The 20 prompts are not all the same kind of question. They deliberately cover the shapes that
real prior-art search takes, including one category the system is expected to *decline*:

| Category | Prompts | What it tests |
|---|---:|---|
| `vague_search` | 7 | An underspecified idea, no claim language |
| `aspect_filter_follow_up` | 6 | "Which of those also does X?" — requires the previous working set |
| `combination_exploration` | 3 | Combining features across several retrieved patents |
| `comparison_follow_up` | 2 | Direct comparison between two previously retrieved patents |
| `similar_patent_search` | 1 | "More like this one" |
| `out_of_domain_insufficiency` | 1 | A question the index cannot support — the correct answer is to say so |

Eleven of the twenty prompts (`requires_context: true`) are only answerable with the previous
turn's results in hand, which is what makes the context-reuse aspect measurable.

---

## 2. Label rubric

Each answer gets one of three ordered labels per aspect. Scores map top → 1.0, middle → 0.5,
bottom → 0.0, and the per-system score is the mean over the 20 prompts.

### Groundedness

| Label | Meaning |
|---|---|
| **Grounded** | Stays close to the retrieved evidence; does not materially exceed what the visible patent set supports |
| Partially grounded | Directionally aligned, but some ranking, interpretation or specificity is not demonstrated by the retrieved set |
| Ungrounded | Substantively goes beyond, or away from, the evidence |

### Helpfulness

| Label | Meaning |
|---|---|
| **Helpful** | Directly answers the ask, rather than only listing patents or restating the query |
| Partially helpful | Points in the right direction but is incomplete, list-like, or misses the constrained ask |
| Unhelpful | Does not advance the user's question |

### Hallucination

| Label | Meaning |
|---|---|
| **None** | No material fabricated claim beyond the retrieved evidence |
| Minor | Core answer plausible, but with overstatement, unsupported specificity, or weakly evidenced linkage |
| Major | A fabricated patent, mapping, or fact |

Reported as **hallucination-freedom** (none → 1.0) so that higher is better on every axis.

### Context reuse

| Label | Meaning |
|---|---|
| **Good** | Correctly builds on the previous turn's working set |
| Partial | Partially carries context, or rebuilds unnecessarily |
| Poor | Ignores prior context on a prompt that needs it |

Prompts marked `requires_context: false` are scored `n/a` and excluded from that aspect.

---

## 3. Results

### Overall

| System | Grounded | Helpful | Hallucination-free | Context reuse | **Overall** |
|---|---:|---:|---:|---:|---:|
| **Our Agent** | **0.900** | **0.850** | 0.900 | **0.975** | **0.906** |
| RAG only | 0.525 | 0.200 | **0.950** | 0.650 | 0.581 |
| ChatGPT Auto | 0.350 | 0.775 | 0.350 | 0.600 | 0.519 |
| Gemini Fast | 0.200 | 0.550 | 0.175 | 0.475 | 0.350 |

![Overall human-rated answer quality](../results/figures/human_eval_overall.png)

### Label counts

Out of 20 answers per system per aspect:

| System | Grounded (top/mid/bottom) | Helpful | Hallucination-free | Context reuse |
|---|---|---|---|---|
| **Our Agent** | 16 / 4 / 0 | 14 / 6 / 0 | 16 / 4 / 0 | 19 / 1 / 0 |
| RAG only | 5 / 11 / 4 | 0 / 8 / 12 | 19 / 0 / 1 | 9 / 8 / 3 |
| ChatGPT Auto | 1 / 12 / 7 | 11 / 9 / 0 | 1 / 12 / 7 | 10 / 4 / 6 |
| Gemini Fast | 0 / 8 / 12 | 3 / 16 / 1 | 0 / 7 / 13 | 9 / 1 / 10 |

![Label distribution](../results/figures/human_eval_label_distribution.png)

### The trade-off the agent escapes

Read the table by column rather than by row and the structure becomes clear:

- **`RAG only` is safe and useless.** It scores 0.950 on hallucination-freedom — the best of
  any system — because it barely asserts anything. It scores 0.200 on helpfulness, with
  **zero** top labels across all 20 prompts, because a ranked list is not an answer.
- **The general assistants are the mirror image.** `ChatGPT Auto` is genuinely helpful
  (0.775, second best) but grounded on only 1 of 20 answers; it produces confident patent
  landscape claims that the retrieved evidence does not support.
- **Our agent is the only system on both axes at once**: 0.850 helpful *and* 0.900 grounded,
  with no fully ungrounded answer and no major hallucination in the whole set.

Note the honest reading of the hallucination column: `RAG only` beats us, 0.950 to 0.900.
Generating an answer at all costs something. The claim is not that the agent is the safest
possible system — it is that it buys a 4.25× improvement in helpfulness for a 0.05 drop in
hallucination-freedom.

### By prompt category

From [`../results/tables/human_eval_by_category.csv`](../results/tables/human_eval_by_category.csv)
(groundedness / helpfulness):

| Category | Our Agent | RAG only | ChatGPT Auto | Gemini Fast |
|---|---|---|---|---|
| `vague_search` | 0.93 / 0.86 | 0.79 / 0.50 | 0.50 / 0.93 | 0.43 / 0.64 |
| `aspect_filter_follow_up` | **1.00 / 1.00** | 0.42 / 0.00 | 0.17 / 0.58 | 0.00 / 0.50 |
| `combination_exploration` | 0.83 / 0.83 | 0.33 / 0.00 | 0.17 / 0.67 | 0.00 / 0.50 |
| `comparison_follow_up` | 0.75 / 0.75 | 0.25 / 0.00 | 0.75 / 1.00 | 0.25 / 0.50 |
| `similar_patent_search` | 0.50 / 0.50 | **1.00** / 0.50 | 0.50 / **1.00** | 0.50 / **1.00** |
| `out_of_domain_insufficiency` | **1.00** / 0.50 | 0.00 / 0.00 | 0.00 / 0.50 | 0.00 / 0.00 |

Three things worth naming:

1. **Follow-ups are where the agent separates.** On `aspect_filter_follow_up` it scores a
   perfect 1.00 on every aspect while the general assistants sit at 0.17 and 0.00 groundedness.
   This is the working-set reuse path doing its job.
2. **Out-of-domain is where grounding is worth the most.** On the prompt the index cannot
   support, the agent is the only system that says so — every other system scores 0.00
   groundedness by confabulating an answer.
3. **`similar_patent_search` is our weakest category** (0.50 / 0.50, behind `RAG only` on
   groundedness). It is a single prompt, so this is a hint rather than a finding, but
   "more like this one" is a query shape the planner does not currently handle specially.

---

## 4. Limitations

- **Small.** 20 prompts, 80 answers. Category-level cells contain as few as one prompt, so
  per-category numbers are directional only.
- **Internally annotated.** Labels come from the project team, not from independent patent
  professionals, and the team knew which system produced which answer.
- **Single domain.** The prompts cluster around one PANORAMA topic area (event-participant
  information systems), so coverage of patent subject matter is narrow.
- **Baselines are not tuned.** `ChatGPT Auto` and `Gemini Fast` were used as a normal user
  would use them, with no prompt engineering to improve their grounding.

---

## 5. Reproducing

```bash
# 1. run the 20 prompts through our agent and the RAG baseline
python -m src.run_product_qa_eval

# 2. build the long-form annotation sheet
python -m src.build_product_qa_manual_eval_sheet

# 3. regenerate the derived tables and figures from the annotated workbook
python tools/make_results_tables.py
python tools/make_figures.py
```

Assistant answers for the two general systems were collected by hand and are stored in
[`eval/product_qa_eval_chatgpt_auto_prefilled.csv`](eval/product_qa_eval_chatgpt_auto_prefilled.csv)
and [`eval/product_qa_eval_gemini_fast_prefilled.csv`](eval/product_qa_eval_gemini_fast_prefilled.csv).
The completed annotations are in
[`eval/product_qa_manual_eval_annotated.xlsx`](eval/product_qa_manual_eval_annotated.xlsx).
