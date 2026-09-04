# Slides

`patent_prior_art_agent.pptx` — the 5-slide project deck (16:9, 12192×6858 EMU).

| # | Section | Contents |
|---|---|---|
| 1 | Research motivation | The gap between benchmarked patent IR and an inspectable prior-art assistant, with citations |
| 2 | Project overview | The six-stage pipeline and what makes it different from single-prompt generation |
| 3 | Impacts | Target users and the six query shapes the agent supports |
| 4 | Validation | Retrieval results on the 3,029-case PAR4PC validation split, plus the three challenges and their fixes |
| 5 | Human evaluation | Head-to-head against ChatGPT Auto, Gemini Fast and RAG-only on 20 prompts |

The numbers on slides 4 and 5 are the same ones in
[`../results/tables/`](../results/tables) and
[`../docs/EVALUATION_RESULTS.md`](../docs/EVALUATION_RESULTS.md). If those change, regenerate
the figures with `make figures` and update the deck.
