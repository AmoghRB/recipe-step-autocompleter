"""Recipe Step Autocompleter: an n-gram language model for recipe instructions.

Built milestone by milestone (see README.md). Standard library only: no neural network.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "recipes.txt"

# Chosen with `python -m app.cli evaluate` (held-out perplexity, see README.md):
# trigram + add-0.01 ties the best bigram on perplexity and writes more fluent steps.
DEFAULT_N, DEFAULT_K = 3, 0.01


# --- Milestone 1: Compile a corpus of cooking recipe instructions.
# Raw recipe text is messy: numbered steps, "Step 2:" labels, bullets, unicode fractions,
# degree signs and "10–12" ranges. We clean each line, split it into sentences (one
# instruction each) and turn every sentence into a list of lowercase word tokens.

_FRACTIONS = {"½": " 1/2", "¼": " 1/4", "¾": " 3/4", "⅓": " 1/3", "⅔": " 2/3"}
_STEP_LABEL = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]?|\d+[.)](?!\d)|[-*•])\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"\d+(?:[./]\d+)?|[a-z]+(?:['-][a-z]+)*|[,;:()]")


def clean_text(text: str) -> str:
    """Normalise one raw recipe line: drop step labels, spell out symbols, lowercase, tidy spaces."""
    for fraction, plain in _FRACTIONS.items():
        text = text.replace(fraction, plain)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-").replace("°", " degrees ")
    text = _STEP_LABEL.sub("", text)
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1 to \2", text)  # "10-12 minutes" -> "10 to 12 minutes"
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    """Clean a piece of text and split it into word, number and punctuation tokens."""
    return _TOKEN.findall(clean_text(text))


def split_sentences(line: str) -> list[str]:
    """One recipe line can hold several instructions; give each sentence back on its own."""
    return [part for part in _SENTENCE_END.split(clean_text(line)) if part]


def load_corpus(path: str | Path = CORPUS_PATH) -> list[list[str]]:
    """Read the raw recipe file and return one token list per instruction sentence.

    Blank lines and lines starting with '#' (recipe titles, comments) are skipped, and
    sentences shorter than two tokens are dropped because they carry no word-to-word signal.
    """
    sentences = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for sentence in split_sentences(line):
            tokens = tokenize(sentence)
            if len(tokens) >= 2:
                sentences.append(tokens)
    return sentences


def split_corpus(sentences: list[list[str]], held_out: float = 0.1, seed: int = 0):
    """Shuffle and split into (train, held_out) so the model is scored on text it never saw."""
    if not 0 < held_out < 1:
        raise ValueError("held_out must be between 0 and 1")
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * held_out))
    return shuffled[cut:], shuffled[:cut]


def corpus_stats(sentences: list[list[str]]) -> dict:
    """Size of the corpus: sentences, tokens and distinct words."""
    return {"sentences": len(sentences),
            "tokens": sum(len(s) for s in sentences),
            "vocabulary": len({w for s in sentences for w in s})}


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
# An n-gram model predicts each word from the n-1 words before it (its context). We count
# how often each word follows each context, and maximum likelihood turns counts into
# probabilities:      P(w | ctx) = count(ctx, w) / count(ctx)
# MLE gives exactly zero to anything unseen, and a sentence's probability is a product, so
# one zero makes the whole sentence impossible and perplexity infinite. Add-k smoothing
# moves a little probability to every word:
#                     P(w | ctx) = (count(ctx, w) + k) / (count(ctx) + k * V)
# If the whole context never appeared in training we back off to the longest shorter
# context that did, so a new trigram still gets a bigram estimate instead of a flat 1/V.

class NGramModel:
    """Word n-gram language model with add-k smoothing and back-off to shorter contexts."""

    def __init__(self, n: int = DEFAULT_N, k: float = DEFAULT_K):
        if n < 1:
            raise ValueError("n must be at least 1")
        if k < 0:
            raise ValueError("k cannot be negative")
        self.n, self.k = n, k
        self.counts: dict[tuple[str, ...], Counter] = {}  # context -> counts of the next word
        self.totals: dict[tuple[str, ...], int] = {}      # context -> how often it was seen
        self.vocab: set[str] = set()

    def fit(self, sentences: list[list[str]]) -> "NGramModel":
        """Count every word after every context of length 0 .. n-1 (so back-off has data)."""
        counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for sentence in sentences:
            padded = [BOS] * (self.n - 1) + list(sentence) + [EOS]
            for i in range(self.n - 1, len(padded)):
                for order in range(self.n):
                    counts[tuple(padded[i - order:i])][padded[i]] += 1
        self.counts = dict(counts)
        self.totals = {ctx: sum(c.values()) for ctx, c in self.counts.items()}
        self.vocab = {w for s in sentences for w in s} | {EOS, UNK}
        return self

    def context(self, history: list[str]) -> tuple[str, ...]:
        """The last n-1 tokens before the next word, padded with <s>; unseen words become <unk>."""
        if self.n == 1:
            return ()
        recent = [w if w in self.vocab else UNK for w in list(history)[-(self.n - 1):]]
        return tuple([BOS] * (self.n - 1 - len(recent)) + recent)

    def backoff(self, context: tuple[str, ...]) -> tuple[str, ...]:
        """The longest ending of `context` that appeared in training (the empty one always did)."""
        for start in range(len(context) + 1):
            if context[start:] in self.totals:
                return context[start:]
        return ()

    def count(self, context: tuple[str, ...], word: str) -> int:
        return self.counts.get(tuple(context), {}).get(word, 0)

    def prob(self, word: str, history: list[str] = ()) -> float:
        """Smoothed P(word | the words before it)."""
        if not self.totals:
            raise RuntimeError("fit the model before asking for probabilities")
        ctx = self.backoff(self.context(history))
        word = word if word in self.vocab else UNK
        seen = self.counts[ctx].get(word, 0)
        return (seen + self.k) / (self.totals[ctx] + self.k * len(self.vocab))

    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp(average negative log-probability per token). Lower is better; inf if any p = 0."""
        log_prob, n_tokens = 0.0, 0
        for sentence in sentences:
            history: list[str] = []
            for word in list(sentence) + [EOS]:
                p = self.prob(word, history)
                if p == 0.0:
                    return math.inf
                log_prob += math.log(p)
                n_tokens += 1
                history.append(word)
        return math.exp(-log_prob / n_tokens)

    def to_dict(self) -> dict:
        return {"n": self.n, "k": self.k, "vocab": sorted(self.vocab),
                "counts": [[list(ctx), dict(c)] for ctx, c in self.counts.items()]}

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(data["n"], data["k"])
        model.counts = {tuple(ctx): Counter(c) for ctx, c in data["counts"]}
        model.totals = {ctx: sum(c.values()) for ctx, c in model.counts.items()}
        model.vocab = set(data["vocab"])
        return model


def evaluate(train: list[list[str]], held_out: list[list[str]],
             orders=(1, 2, 3, 4), ks=(0.0, 0.01, 0.1, 1.0)) -> list[dict]:
    """Held-out perplexity for every (n, k) pair: the honest way to choose a model."""
    return [{"n": n, "k": k, "perplexity": NGramModel(n, k).fit(train).perplexity(held_out)}
            for n in orders for k in ks]


# --- Milestone 3: Create a function to generate recipe step continuations.
# Hint: Develop a function that uses your trained n-gram model to predict subsequent words.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Build a CLI for users to input partial steps and get suggestions.
# Hint: Construct a command-line interface to take user input and display predictions.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
