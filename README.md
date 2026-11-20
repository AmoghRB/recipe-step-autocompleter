# Recipe Step Autocompleter

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/xs5cac6k4m/badge.svg)](https://msrit.ask.thili.ai/b/xs5cac6k4m)
[![tests](https://github.com/AmoghRB/recipe-step-autocompleter/actions/workflows/ci.yml/badge.svg)](https://github.com/AmoghRB/recipe-step-autocompleter/actions/workflows/ci.yml)

A command-line tool for home cooks. Type the start of a recipe step and it suggests the next
word and a few ways to finish the step, so you don't miss anything. It is a word **n-gram
language model** built from scratch in plain Python (standard library only, no neural
network), trained on a corpus of recipe instructions.

```text
$ python -m app.cli suggest heat 2 tablespoons of
Next word:  oil 30%  ·  butter 13%  ·  ghee 13%  ·  cream 7%  ·  cocoa 3%
Finish the step:
  1. Heat 2 tablespoons of oil in a pan over medium heat.
  2. Heat 2 tablespoons of butter in a pan over medium heat.
  3. Heat 2 tablespoons of ghee in a pan over medium heat.
```

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli train          # train, report held-out perplexity, save the model
python -m app.cli interactive    # type partial steps one after another
pytest
```

No API key is needed: the model is the n-gram code in `app/core.py`, so `.env` can stay
empty. (`app/llm.py` came with the starter and is not used.)

## Commands

| Command | What it does |
|---|---|
| `suggest <words…>` | Likely next words with probabilities, plus full-step endings |
| `interactive` | A prompt (`step>`): type a partial step, get suggestions, empty line quits |
| `generate [<words…>]` | Writes whole steps by sampling (`--count`, `--seed`, `--temperature`) |
| `train` | Trains on `corpus/recipes.txt`, prints held-out perplexity, saves the model |
| `evaluate` | Held-out perplexity for n = 1–4 and k = 0, 0.01, 0.1, 1 |

Options for `suggest` and `interactive`: `--top N` next words, `--num N` endings,
`--only verbs|ingredients` to keep only cooking verbs or ingredients. Every model command
takes `--n`, `--k` and `--retrain`.

```text
$ python -m app.cli interactive --num 2
step> preheat the oven to
Next word:  180 24%  ·  175 8%  ·  190 8%  ·  200 8%  ·  220 8%
Finish the step:
  1. Preheat the oven to 180°C and grease a loaf tin and smooth the top.
  2. Preheat the oven to 175°C and grease a loaf tin and smooth the top.
step> cook on low heat
Next word:  for 39%  ·  (end of step) 20%  ·  until 7%
Finish the step:
  1. Cook on low heat for 5 minutes.
  2. Cook on low heat.
```

## How it works

1. **Corpus** (`corpus/recipes.txt`, `load_corpus`). 40 home-style recipes, Indian and
   everyday Western, written for this project. Cleaning drops step labels (`1.`, `Step 2:`,
   bullets), spells out `½` → `1/2`, `180°C` → `180 degrees c` and `10–12` → `10 to 12`,
   lowercases, then splits each line into sentences and tokens. Result: 366 steps,
   4,410 tokens, 538 distinct words.
2. **Model** (`NGramModel`). Counts every word after every context of 0 to n−1 words and
   scores with add-k smoothed maximum likelihood:
   `P(w | ctx) = (count(ctx, w) + k) / (count(ctx) + k·V)`.
   If a context never appeared in training it backs off to the longest shorter context that
   did, so a new trigram still gets a bigram estimate instead of a flat 1/V.
3. **Continuations** (`next_words`, `complete`, `generate`). Suggestions come from the words
   actually seen after the context, ranked by count. `complete` starts one ending from each
   top next word and extends it greedily, refusing to repeat a trigram so it can't loop.
   `generate` samples instead, with a seed and a temperature. Smoothing is used for scoring
   new text, not for picking words, or random vocabulary would leak into the steps.
4. **CLI** (`app/cli.py`). The trained model is saved to `data/app.json` and reloaded on the
   next run. It retrains only when n, k or the corpus file change.

## Choosing n and k

`python -m app.cli evaluate` trains on 90% of the steps and measures perplexity on the other
10%. Lower is better.

| n \ k | 0 | 0.01 | 0.1 | 1 (Laplace) |
|---|---|---|---|---|
| 1 (unigram) | inf | 192.7 | 175.0 | 160.5 |
| 2 (bigram) | inf | **43.9** | 58.8 | 141.1 |
| 3 (trigram) | inf | 44.6 | 76.1 | 190.1 |
| 4 | inf | 49.5 | 86.6 | 207.4 |

- **k = 0 is always infinite.** One held-out word the model never saw gets probability 0,
  and the product over the whole text becomes 0.
- **Add-1 is far worse than add-0.01.** With 538 words, adding 1 to every count takes most
  of the probability away from the phrases that really occur.
- **Bigram and trigram tie.** The trigram uses more context but most trigrams are rare in a
  corpus this small, so sparsity cancels the gain. The default is **trigram, k = 0.01**:
  the same perplexity, and its completions read better ("heat 2 tablespoons of oil in a pan
  over medium heat").
- Training-text perplexity is much lower than held-out perplexity, which is why only the
  held-out number is reported.

## Tests

`pytest` runs 23 tests, grouped by milestone in `tests/test_core.py`. They cover the
cleaning rules, MLE on a hand-checkable corpus (`P(york | new) = 40/100`), zeros with and
without smoothing, every distribution summing to 1, the perplexity comparisons above,
loop-free completions, the saved model being reloaded, and each CLI command. GitHub Actions
runs them on every push.

## Milestones

- [x] **1. Compile a corpus of cooking recipe instructions.**
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities with smoothing.**
- [x] **3. Create a function to generate recipe step continuations.**
- [x] **4. Build a CLI for users to input partial steps and get suggestions.**

Stretch ideas, all done: choose the number of suggestions (`--top`, `--num`), save and load
the trained model, filter suggestions to cooking verbs or ingredients (`--only`).

## Limits

- 366 steps is small. Long completions sometimes join two recipes ("…grease a loaf tin and
  smooth the top"), which is a known n-gram weakness: the model only ever sees two words back.
- Quantities are copied from the corpus, not reasoned about.
- Built with help from an AI coding assistant (Claude).
