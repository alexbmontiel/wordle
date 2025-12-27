import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import cpu_count
from tqdm import tqdm


@lru_cache(maxsize=500000)
def compute_result(guess: str, answer: str) -> str:
    """
    Compute the G/Y/N result string for a guess against an answer.
    Cached for performance.
    """
    result = ["N"] * 5
    answer_chars: list[str | None] = list(answer)

    # First pass: mark greens and remove matched chars from answer
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = "G"
            answer_chars[i] = None

    # Second pass: mark yellows
    for i, g in enumerate(guess):
        if result[i] == "G":
            continue
        if g in answer_chars:
            result[i] = "Y"
            answer_chars[answer_chars.index(g)] = None

    return "".join(result)


def partition_by_result(
    guess: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
) -> dict[str, tuple[list[str], float]]:
    """
    Partition words by what result they'd produce for this guess.

    Returns a dict mapping result string -> (list of words, sum of frequencies).
    At most 243 buckets (3^5 possible results).
    """
    buckets: dict[str, tuple[list[str], float]] = {}
    for word, (freq, _) in word_list.items():
        result = compute_result(guess, word)
        if result in buckets:
            words, freq_sum = buckets[result]
            words.append(word)
            buckets[result] = (words, freq_sum + freq)
        else:
            buckets[result] = ([word], freq)
    return buckets


def expected_remaining(
    guess: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
) -> float:
    """
    Score a guess by expected remaining words after guessing (frequency-weighted).

    Lower is better - measures expected remaining probability mass.
    """
    buckets = partition_by_result(guess, word_list)
    total_freq = sum(data[0] for data in word_list.values())

    if total_freq == 0:
        return 0.0

    # Expected value weighted by frequency
    # sum of (p_bucket * freq_bucket) where p_bucket = freq_bucket / total_freq
    return sum(freq_sum ** 2 for _, freq_sum in buckets.values()) / total_freq


def information_gain(
    guess: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
) -> float:
    """
    Calculate expected information gain (entropy) for a guess, weighted by word frequency.

    Higher is better - measures how much uncertainty is reduced on average,
    giving more weight to eliminating common words.
    """
    buckets = partition_by_result(guess, word_list)
    total_freq = sum(data[0] for data in word_list.values())

    if total_freq <= 0:
        return 0.0

    # Entropy weighted by frequency: sum(p * log2(1/p))
    # where p = bucket_freq_sum / total_freq
    entropy = 0.0
    for _, freq_sum in buckets.values():
        p = freq_sum / total_freq
        if p > 0:
            entropy += p * math.log2(1 / p)

    return entropy


def rank_guesses(
    word_list: dict[str, tuple[float, frozenset[str]]],
    candidates: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Rank guesses by information gain (highest first).

    Args:
        word_list: Current possible words
        candidates: Words to evaluate (defaults to all words in word_list)

    Returns:
        List of (word, score) tuples sorted by score descending
    """
    if candidates is None:
        candidates = list(word_list.keys())

    scored = [(word, information_gain(word, word_list)) for word in tqdm(candidates)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def best_guess(word_list: dict[str, tuple[float, frozenset[str]]]) -> str | None:
    """Return the best guess for the current word list."""
    if not word_list:
        return None
    if len(word_list) == 1:
        return next(iter(word_list))

    best_word = None
    best_score = -1.0
    for word in word_list:
        score = information_gain(word, word_list)
        if score > best_score:
            best_score = score
            best_word = word
    return best_word


def play_game(
    answer: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
    starting_word: str,
    max_guesses: int = 6,
) -> int:
    """
    Play a game against a known answer.

    Returns the number of guesses taken (or max_guesses + 1 if failed).
    """
    from wordle_helper.filter import filter_word_list

    remaining = word_list.copy()
    guess = starting_word

    for turn in range(1, max_guesses + 1):
        if guess == answer:
            return turn

        result = compute_result(guess, answer)
        remaining = filter_word_list(guess, result, remaining)

        if not remaining:
            return max_guesses + 1  # No valid words left (shouldn't happen)

        guess = best_guess(remaining)
        if guess is None:
            return max_guesses + 1

    # Check if last guess was correct
    if guess == answer:
        return max_guesses

    return max_guesses + 1  # Failed


def evaluate_starting_word(
    starting_word: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
    answers: list[str] | None = None,
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
    show_progress: bool = True,
) -> dict:
    """
    Evaluate a starting word by playing against a set of answers.

    Args:
        starting_word: The word to evaluate
        word_list: All possible words (used for guessing during game)
        answers: List of answers to test against (defaults to all words in word_list)
        max_guesses: Maximum allowed guesses
        fail_penalty: Score to assign for failures (default 10)
        show_progress: Whether to show progress bar

    Returns:
        Dict with statistics:
        - starting_word: The evaluated word
        - average_score: Mean guesses (failures count as fail_penalty)
        - failure_rate: Fraction of games that failed
        - distribution: Dict mapping guess_count -> number of games
        - total_games: Number of games played
    """
    if answers is None:
        answers = list(word_list.keys())

    distribution: dict[int, int] = {}
    total_score = 0.0
    failures = 0

    iterator = tqdm(answers, desc=f"Evaluating {starting_word}") if show_progress else answers
    for answer in iterator:
        guesses = play_game(answer, word_list, starting_word, max_guesses)

        if guesses > max_guesses:
            failures += 1
            total_score += fail_penalty
            distribution[max_guesses + 1] = distribution.get(max_guesses + 1, 0) + 1
        else:
            total_score += guesses
            distribution[guesses] = distribution.get(guesses, 0) + 1

    total_games = len(answers)
    return {
        "starting_word": starting_word,
        "average_score": total_score / total_games if total_games > 0 else 0,
        "failure_rate": failures / total_games if total_games > 0 else 0,
        "distribution": dict(sorted(distribution.items())),
        "total_games": total_games,
    }


def rank_starting_words(
    word_list: dict[str, tuple[float, frozenset[str]]],
    candidates: list[str] | None = None,
    answers: list[str] | None = None,
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
) -> list[dict]:
    """
    Rank starting words by expected number of guesses.

    Args:
        word_list: All possible words (used for guessing during game)
        candidates: Starting words to evaluate (defaults to all words in word_list)
        answers: List of answers to test against (defaults to all words in word_list)
        max_guesses: Maximum allowed guesses
        fail_penalty: Score to assign for failures

    Returns:
        List of result dicts sorted by average_score (ascending = better)
    """
    if candidates is None:
        candidates = list(word_list.keys())

    if answers is None:
        answers = list(word_list.keys())

    results = []
    for starting_word in tqdm(candidates, desc="Ranking starting words"):
        result = evaluate_starting_word(
            starting_word=starting_word,
            word_list=word_list,
            answers=answers,
            max_guesses=max_guesses,
            fail_penalty=fail_penalty,
            show_progress=False,
        )
        results.append(result)

    # Sort by average_score (lower is better)
    results.sort(key=lambda x: x["average_score"])
    return results


def _evaluate_worker(args: tuple) -> dict:
    """Worker function for parallel evaluation."""
    starting_word, word_list, answers, max_guesses, fail_penalty = args
    return evaluate_starting_word(
        starting_word, word_list, answers, max_guesses, fail_penalty, show_progress=False
    )


def rank_starting_words_parallel(
    word_list: dict[str, tuple[float, frozenset[str]]],
    candidates: list[str] | None = None,
    answers: list[str] | None = None,
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
    n_workers: int | None = None,
) -> list[dict]:
    """
    Rank starting words using parallel processing (optimal strategy).

    Args:
        word_list: All possible words (used for guessing during game)
        candidates: Starting words to evaluate (defaults to all words in word_list)
        answers: List of answers to test against (defaults to all words in word_list)
        max_guesses: Maximum allowed guesses
        fail_penalty: Score to assign for failures
        n_workers: Number of parallel workers (defaults to CPU count)

    Returns:
        List of result dicts sorted by average_score (ascending = better)
    """
    if candidates is None:
        candidates = list(word_list.keys())

    if answers is None:
        answers = list(word_list.keys())

    if n_workers is None:
        n_workers = cpu_count()

    # Prepare arguments for workers
    args_list = [
        (starting_word, word_list, answers, max_guesses, fail_penalty)
        for starting_word in candidates
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_evaluate_worker, args): args[0] for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Ranking (parallel)"):
            result = future.result()
            results.append(result)

    # Sort by average_score (lower is better)
    results.sort(key=lambda x: x["average_score"])
    return results