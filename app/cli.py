"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app import core

FILTERS = {"verbs": core.COOKING_VERBS, "ingredients": core.INGREDIENTS}


def _model(args) -> core.NGramModel:
    return core.get_model(n=args.n, k=args.k, retrain=args.retrain)


def _train(args) -> None:
    sentences = core.load_corpus()
    stats = core.corpus_stats(sentences)
    train, held = core.split_corpus(sentences)
    score = core.NGramModel(args.n, args.k).fit(train).perplexity(held)
    core.get_model(n=args.n, k=args.k, retrain=True)
    print(f"Corpus: {stats['sentences']} steps, {stats['tokens']} tokens, {stats['vocabulary']} distinct words.")
    print(f"{args.n}-gram model, add-{args.k} smoothing: held-out perplexity {score:.1f} "
          f"(trained on 90%, scored on the other 10%).")
    print("Trained on the full corpus and saved, so later commands load it instead of retraining.")


def _evaluate(_args) -> None:
    train, held = core.split_corpus(core.load_corpus())
    rows = core.evaluate(train, held)
    ks = sorted({r["k"] for r in rows})
    print("Held-out perplexity (lower is better)\n")
    print("n \\ k   " + "".join(f"{k:>10}" for k in ks))
    for n in sorted({r["n"] for r in rows}):
        cells = [next(r["perplexity"] for r in rows if r["n"] == n and r["k"] == k) for k in ks]
        print(f"{n:<8}" + "".join(f"{p:>10.1f}" for p in cells))
    best = min(rows, key=lambda r: r["perplexity"])
    print(f"\nBest: n={best['n']}, k={best['k']} ({best['perplexity']:.1f}). "
          "k=0 is infinite: one unseen word makes the whole text impossible.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recipe-step-autocompleter", description="Home cooks use this CLI tool to autocomplete incomplete recipe instructions, ensuring they don't miss steps and can follow recipes smoothly.")
    model_opts = argparse.ArgumentParser(add_help=False)
    model_opts.add_argument("--n", type=int, default=core.DEFAULT_N, help="n-gram order (default: %(default)s)")
    model_opts.add_argument("--k", type=float, default=core.DEFAULT_K, help="add-k smoothing (default: %(default)s)")
    model_opts.add_argument("--retrain", action="store_true", help="ignore the saved model and train again")
    suggest_opts = argparse.ArgumentParser(add_help=False)
    suggest_opts.add_argument("--top", type=int, default=5, help="how many next words to show (default: %(default)s)")
    suggest_opts.add_argument("--num", type=int, default=3, help="how many full-step endings (default: %(default)s)")
    suggest_opts.add_argument("--only", choices=sorted(FILTERS), help="only suggest cooking verbs or ingredients")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("train", parents=[model_opts], help="train on corpus/recipes.txt, report perplexity, save the model")
    p = sub.add_parser("suggest", parents=[model_opts, suggest_opts], help="suggestions for one partial step")
    p.add_argument("text", nargs="+", help="the start of a step, e.g. heat 2 tablespoons of")
    sub.add_parser("interactive", parents=[model_opts, suggest_opts], help="type partial steps one after another")
    p = sub.add_parser("generate", parents=[model_opts], help="write whole steps by sampling the model")
    p.add_argument("text", nargs="*", help="optional start of the step")
    p.add_argument("--count", type=int, default=1, help="how many steps (default: %(default)s)")
    p.add_argument("--seed", type=int, help="make the output repeatable")
    p.add_argument("--temperature", type=float, default=1.0, help="below 1 is safer, above 1 is wilder")
    sub.add_parser("evaluate", help="compare held-out perplexity for different n and k")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Recipe Step Autocompleter is ready. Open README.md for your milestones.")
    elif args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "suggest":
        print(core.format_suggestions(_model(args), " ".join(args.text), top=args.top, num=args.num,
                                      only=FILTERS.get(args.only)))
    elif args.command == "interactive":
        core.run_interactive(_model(args), top=args.top, num=args.num, only=FILTERS.get(args.only))
    elif args.command == "generate":
        model = _model(args)
        for i in range(args.count):
            seed = None if args.seed is None else args.seed + i
            print(core.generate(model, " ".join(args.text), seed=seed, temperature=args.temperature))


if __name__ == "__main__":
    main()
