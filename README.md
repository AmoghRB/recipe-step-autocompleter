# Recipe Step Autocompleter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/xs5cac6k4m/badge.svg)](https://ask.thili.ai/b/xs5cac6k4m)

This Recipe Step Autocompleter is a CLI tool designed for home cooks. It helps complete partial recipe instructions, preventing missed steps and ensuring a smooth cooking experience. It leverages next-word prediction and text generation capabilities to suggest logical continuations for recipe instructions.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps home cooks autocomplete incomplete recipe instructions, ensuring they follow recipes smoothly.

**Done when.** The tool successfully compiles a corpus of cooking recipe instructions, implements n-gram counting with smoothing, generates recipe step continuations, and provides a functional CLI for users to get suggestions for partial steps.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Compile a corpus of cooking recipe instructions.**  
      Hint: Use text processing techniques to clean and prepare your recipe data.
- [ ] **2. Implement n-gram counting and maximum-likelihood probabilities with smoothing.**  
      Hint: Apply n-gram counting and maximum-likelihood estimation, incorporating smoothing techniques.
- [ ] **3. Create a function to generate recipe step continuations.**  
      Hint: Develop a function that uses your trained n-gram model to predict subsequent words.
- [ ] **4. Build a CLI for users to input partial steps and get suggestions.**  
      Hint: Construct a command-line interface to take user input and display predictions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the number of suggested next words or phrases.
- Implement a feature to save and load trained models to avoid retraining on every run.
- Add an option to filter suggestions based on common cooking verbs or ingredients.
