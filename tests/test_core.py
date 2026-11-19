"""Tests for each milestone in README.md."""
import math

import pytest

from app import core


# Milestone 1: Compile a corpus of cooking recipe instructions.
def test_milestone_1_cleaning_removes_labels_and_spells_out_symbols():
    assert core.clean_text("Step 3:  Bake at 180°C for 10–12 minutes.") == \
        "bake at 180 degrees c for 10 to 12 minutes."
    assert core.clean_text("2. Add ½ cup of milk") == "add 1/2 cup of milk"
    assert core.clean_text("- Whisk the EGGS") == "whisk the eggs"


def test_milestone_1_tokenize_keeps_words_numbers_and_commas():
    assert core.tokenize("Add 1 ½ cups of flour, then mix.") == \
        ["add", "1", "1/2", "cups", "of", "flour", ",", "then", "mix"]
    assert core.tokenize("Heat a non-stick pan") == ["heat", "a", "non-stick", "pan"]


def test_milestone_1_corpus_is_clean_and_big_enough():
    sentences = core.load_corpus()
    stats = core.corpus_stats(sentences)
    assert stats["sentences"] > 300 and stats["vocabulary"] > 300
    # No titles, step labels or raw symbols survive cleaning.
    flat = {w for s in sentences for w in s}
    assert "step" not in flat and "°" not in "".join(flat) and "½" not in "".join(flat)
    assert ["preheat", "the", "oven", "to", "200", "degrees", "c"] in sentences


def test_milestone_1_split_is_repeatable_and_disjoint():
    sentences = core.load_corpus()
    train, held = core.split_corpus(sentences, held_out=0.1, seed=0)
    assert len(train) + len(held) == len(sentences)
    assert core.split_corpus(sentences, 0.1, 0) == (train, held)


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
def _new_york_corpus():
    """'new' appears 100 times, 'new york' 40 times."""
    return [["new", "york"]] * 40 + [["new", "delhi"]] * 60


def test_milestone_2_mle_is_pair_count_over_context_count():
    model = core.NGramModel(n=2, k=0).fit(_new_york_corpus())
    assert model.count(("new",), "york") == 40
    assert model.prob("york", ["new"]) == pytest.approx(40 / 100)


def test_milestone_2_add_k_smoothing_removes_zeros():
    unseen_pair = ("delhi", ["york"])  # 'york delhi' never occurs
    assert core.NGramModel(n=2, k=0).fit(_new_york_corpus()).prob(*unseen_pair) == 0.0
    smoothed = core.NGramModel(n=2, k=1).fit(_new_york_corpus())
    vocab_size = len(smoothed.vocab)  # new, york, delhi, </s>, <unk>
    assert smoothed.prob(*unseen_pair) == pytest.approx(1 / (40 + vocab_size))


def test_milestone_2_every_distribution_sums_to_one():
    model = core.NGramModel(3, 0.01).fit(core.load_corpus())
    for history in ([], ["preheat", "the"], ["never", "seen"]):
        assert sum(model.prob(w, history) for w in model.vocab) == pytest.approx(1.0)


def test_milestone_2_unseen_context_backs_off_to_a_shorter_one():
    model = core.NGramModel(3, 0.01).fit(core.load_corpus())
    assert model.backoff(model.context(["zebra", "the"])) == ("the",)


def test_milestone_2_held_out_perplexity_picks_the_model():
    train, held = core.split_corpus(core.load_corpus())
    assert core.NGramModel(3, k=0).fit(train).perplexity(held) == math.inf  # one zero is fatal
    chosen = core.NGramModel(3, 0.01).fit(train)
    assert chosen.perplexity(held) < core.NGramModel(1, 0.01).fit(train).perplexity(held)
    assert chosen.perplexity(held) < core.NGramModel(3, 1.0).fit(train).perplexity(held)  # add-1
    assert chosen.perplexity(train) < chosen.perplexity(held)  # training text flatters the model


# Milestone 3: Create a function to generate recipe step continuations.
@pytest.fixture(scope="module")
def recipe_model():
    return core.NGramModel().fit(core.load_corpus())


def test_milestone_3_complete_ranks_continuations_by_count():
    toy = [["heat", "the", "oil"]] * 2 + [["heat", "the", "pan"]]
    model = core.NGramModel(n=3, k=0.01).fit(toy)
    assert core.complete(model, "Heat the", num=2) == ["Heat the oil.", "Heat the pan."]


def test_milestone_3_next_words_are_ranked_probabilities(recipe_model):
    suggestions = core.next_words(recipe_model, "preheat the", top=3)
    assert suggestions[0][0] == "oven"
    probs = [p for _, p in core.next_words(recipe_model, "heat 2 tablespoons of", top=5)]
    assert probs == sorted(probs, reverse=True) and all(0 < p < 1 for p in probs)


def test_milestone_3_filters_keep_only_verbs_or_ingredients(recipe_model):
    words = [w for w, _ in core.next_words(recipe_model, "chopped", top=5, only=core.INGREDIENTS)]
    assert words and set(words) <= core.INGREDIENTS
    verbs = [w for w, _ in core.next_words(recipe_model, "", top=5, only=core.COOKING_VERBS)]
    assert verbs and set(verbs) <= core.COOKING_VERBS


def test_milestone_3_generate_is_repeatable_and_clean(recipe_model):
    first = core.generate(recipe_model, "add the", seed=3)
    assert first == core.generate(recipe_model, "add the", seed=3)
    assert first.startswith("Add the") and first.endswith(".")
    assert "<s>" not in first and "</s>" not in first and "<unk>" not in first


def test_milestone_3_greedy_completion_never_loops(recipe_model):
    for step in core.complete(recipe_model, "stir", num=3, max_words=40):
        tokens = core.tokenize(step)
        trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
        assert len(trigrams) == len(set(trigrams))


def test_milestone_3_detokenize_reads_like_a_recipe():
    tokens = ["preheat", "the", "oven", "to", "180", "degrees", "c", ",", "then", "bake"]
    assert core.detokenize(tokens) == "Preheat the oven to 180°C, then bake."


# Milestone 4: Build a CLI for users to input partial steps and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
