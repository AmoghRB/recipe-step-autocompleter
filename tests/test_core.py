"""Tests for each milestone in README.md."""
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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Create a function to generate recipe step continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a CLI for users to input partial steps and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
