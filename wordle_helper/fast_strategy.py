"""
Fast strategy evaluation using pre-computed result matrices.
Optimized for evaluating many starting words.
"""
import numpy as np
from numba import njit, prange
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, Pool


# Encode results as integers 0-242 (3^5 - 1)
RESULT_TO_INT = {}
INT_TO_RESULT = {}
for i in range(243):
    result = ""
    val = i
    for _ in range(5):
        result += "GYN"[val % 3]
        val //= 3
    RESULT_TO_INT[result] = i
    INT_TO_RESULT[i] = result


@njit(cache=True)
def _compute_entropy(results: np.ndarray, freqs: np.ndarray, total_freq: float) -> float:
    """Compute entropy for a single guess (numba-optimized)."""
    # Aggregate frequencies by result bucket
    freq_per_result = np.zeros(243, dtype=np.float64)
    for i in range(len(results)):
        freq_per_result[results[i]] += freqs[i]

    # Compute entropy
    entropy = 0.0
    for f in freq_per_result:
        if f > 0:
            p = f / total_freq
            entropy -= p * np.log2(p)
    return entropy


@njit(cache=True)
def _find_best_guess(result_submatrix: np.ndarray, freqs: np.ndarray, total_freq: float) -> int:
    """Find the best guess index using JIT-compiled entropy computation."""
    n = result_submatrix.shape[0]
    best_idx = 0
    best_score = -1.0

    for i in range(n):
        score = _compute_entropy(result_submatrix[i], freqs, total_freq)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


@njit(cache=True)
def _compute_result_numba(guess_arr: np.ndarray, answer_arr: np.ndarray) -> int:
    """Numba-optimized result computation for single guess/answer pair."""
    guess_results = np.full(5, 2, dtype=np.int8)  # 2 = N, 1 = Y, 0 = G
    answer_used = np.zeros(5, dtype=np.bool_)

    # First pass: greens
    for i in range(5):
        if guess_arr[i] == answer_arr[i]:
            guess_results[i] = 0  # G
            answer_used[i] = True

    # Second pass: yellows
    for i in range(5):
        if guess_results[i] == 0:
            continue
        for j in range(5):
            if not answer_used[j] and guess_arr[i] == answer_arr[j]:
                guess_results[i] = 1  # Y
                answer_used[j] = True
                break

    # Encode as integer
    result = 0
    multiplier = 1
    for i in range(5):
        result += guess_results[i] * multiplier
        multiplier *= 3

    return result


@njit(cache=True, parallel=True)
def _compute_row_numba(guess_arr: np.ndarray, all_answers: np.ndarray) -> np.ndarray:
    """Compute results for one guess against all answers (numba parallel)."""
    n = all_answers.shape[0]
    row = np.zeros(n, dtype=np.uint8)
    for j in prange(n):
        row[j] = _compute_result_numba(guess_arr, all_answers[j])
    return row


@njit(cache=True, parallel=True)
def _compute_full_matrix_numba(all_words: np.ndarray) -> np.ndarray:
    """Compute full result matrix using numba with parallelization."""
    n = all_words.shape[0]
    matrix = np.zeros((n, n), dtype=np.uint8)
    for i in prange(n):
        for j in range(n):
            matrix[i, j] = _compute_result_numba(all_words[i], all_words[j])
    return matrix


def words_to_array(words: list[str]) -> np.ndarray:
    """Convert list of words to numpy array of ordinals for numba."""
    n = len(words)
    arr = np.zeros((n, 5), dtype=np.int32)
    for i, word in enumerate(words):
        for j, c in enumerate(word):
            arr[i, j] = ord(c)
    return arr


def compute_result_fast(guess: str, answer: str) -> int:
    """Compute result as integer (0-242)."""
    result = 0
    multiplier = 1
    answer_chars = list(answer)
    guess_results = [2] * 5  # 2 = N, 1 = Y, 0 = G

    # First pass: greens
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            guess_results[i] = 0  # G
            answer_chars[i] = None

    # Second pass: yellows
    for i, g in enumerate(guess):
        if guess_results[i] == 0:
            continue
        if g in answer_chars:
            guess_results[i] = 1  # Y
            answer_chars[answer_chars.index(g)] = None

    for r in guess_results:
        result += r * multiplier
        multiplier *= 3

    return result


def _compute_row(args: tuple) -> np.ndarray:
    """Compute one row of the result matrix."""
    guess, words = args
    n = len(words)
    row = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        row[j] = compute_result_fast(guess, words[j])
    return row


# Shared data for parallel workers
_WORKER_DATA: dict = {}


def _init_eval_worker(words, frequencies, result_matrix, answer_indices, max_guesses, fail_penalty):
    """Initialize worker process with shared data."""
    _WORKER_DATA['words'] = words
    _WORKER_DATA['frequencies'] = frequencies
    _WORKER_DATA['result_matrix'] = result_matrix
    _WORKER_DATA['answer_indices'] = answer_indices
    _WORKER_DATA['max_guesses'] = max_guesses
    _WORKER_DATA['fail_penalty'] = fail_penalty


def _evaluate_starting_worker(starting_idx: int) -> dict:
    """Worker function for evaluating a starting word."""
    words = _WORKER_DATA['words']
    frequencies = _WORKER_DATA['frequencies']
    result_matrix = _WORKER_DATA['result_matrix']
    answer_indices = _WORKER_DATA['answer_indices']
    max_guesses = _WORKER_DATA['max_guesses']
    fail_penalty = _WORKER_DATA['fail_penalty']

    n_words = len(words)
    starting_word = words[starting_idx]

    distribution: dict[int, int] = {}
    total_score = 0.0
    failures = 0
    cache: dict[bytes, int] = {}

    for answer_idx in answer_indices:
        # Play game
        remaining_mask = np.ones(n_words, dtype=bool)
        guess_idx = starting_idx

        for turn in range(1, max_guesses + 1):
            if guess_idx == answer_idx:
                guesses = turn
                break

            result = result_matrix[guess_idx, answer_idx]
            remaining_mask = remaining_mask & (result_matrix[guess_idx] == result)

            if not remaining_mask.any():
                guesses = max_guesses + 1
                break

            # Find best guess (with caching)
            cache_key = remaining_mask.tobytes()
            cached = cache.get(cache_key)
            if cached is not None:
                guess_idx = cached
            else:
                remaining_indices = np.where(remaining_mask)[0]
                n_remaining = len(remaining_indices)

                if n_remaining == 1:
                    guess_idx = remaining_indices[0]
                else:
                    remaining_freqs = frequencies[remaining_indices].astype(np.float64)
                    total_freq = remaining_freqs.sum()

                    if total_freq == 0:
                        guess_idx = remaining_indices[0]
                    else:
                        result_submatrix = result_matrix[remaining_indices][:, remaining_indices]
                        best_local_idx = _find_best_guess(result_submatrix, remaining_freqs, total_freq)
                        guess_idx = remaining_indices[best_local_idx]

                cache[cache_key] = guess_idx
        else:
            # Loop completed without break
            if guess_idx == answer_idx:
                guesses = max_guesses
            else:
                guesses = max_guesses + 1

        # Record result
        if guesses > max_guesses:
            failures += 1
            total_score += fail_penalty
            distribution[max_guesses + 1] = distribution.get(max_guesses + 1, 0) + 1
        else:
            total_score += guesses
            distribution[guesses] = distribution.get(guesses, 0) + 1

    total_games = len(answer_indices)
    return {
        "starting_word": starting_word,
        "average_score": total_score / total_games if total_games > 0 else 0,
        "failure_rate": failures / total_games if total_games > 0 else 0,
        "distribution": dict(sorted(distribution.items())),
        "total_games": total_games,
    }


def build_result_matrix(words: list[str], show_progress: bool = True, n_workers: int | None = None, use_numba: bool = True) -> np.ndarray:
    """
    Pre-compute all guess/answer result pairs.

    Returns: numpy array of shape (n_words, n_words) with dtype uint8
    Each entry is the result code (0-242) for result_matrix[guess_idx, answer_idx]

    Args:
        words: List of words
        show_progress: Show progress bar
        n_workers: Number of workers (ignored when use_numba=True)
        use_numba: Use numba-optimized parallel computation (much faster)
    """
    n = len(words)

    if use_numba:
        # Convert words to numpy array for numba
        if show_progress:
            print(f"Converting {n} words to array...")
        words_arr = words_to_array(words)

        if show_progress:
            print("Computing result matrix with numba (parallel)...")

        # First call compiles, subsequent calls are fast
        matrix = _compute_full_matrix_numba(words_arr)

        if show_progress:
            print("Done!")
        return matrix

    # Fallback to ProcessPoolExecutor
    matrix = np.zeros((n, n), dtype=np.uint8)

    if n_workers is None:
        n_workers = cpu_count()

    args_list = [(words[i], words) for i in range(n)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunksize = max(1, n // (n_workers * 4))
        results = executor.map(_compute_row, args_list, chunksize=chunksize)

        iterator = tqdm(results, total=n, desc="Building result matrix") if show_progress else results
        for i, row in enumerate(iterator):
            matrix[i] = row

    return matrix


class FastEvaluator:
    """
    Fast evaluator using pre-computed result matrix.
    """

    def __init__(
        self,
        word_list: dict[str, tuple[float, frozenset[str]]],
        answer_indices: list[int] | None = None,
        result_matrix: np.ndarray | None = None,
    ):
        """
        Initialize with word list.

        Args:
            word_list: Dict mapping word -> (frequency, char_set)
            answer_indices: Indices of words to use as answers (default: all)
            result_matrix: Pre-computed result matrix (optional, builds if not provided)
        """
        self.words = list(word_list.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.frequencies = np.array([word_list[w][0] for w in self.words])
        self.n_words = len(self.words)

        # Build or use provided result matrix
        if result_matrix is not None:
            self.result_matrix = result_matrix
        else:
            self.result_matrix = build_result_matrix(self.words)

        # Answer set
        if answer_indices is None:
            self.answer_indices = np.arange(self.n_words)
        else:
            self.answer_indices = np.array(answer_indices)

        # Pre-compute frequency sums for entropy calculation
        self.total_freq = self.frequencies.sum()

        # Cache for best_guess computations (cleared per evaluation)
        self._best_guess_cache: dict[tuple, int] = {}

    def get_remaining_mask(self, guess_idx: int, result: int, current_mask: np.ndarray) -> np.ndarray:
        """Get mask of words remaining after a guess."""
        # Words where result_matrix[guess_idx, word] == result AND currently valid
        return current_mask & (self.result_matrix[guess_idx] == result)

    def best_guess_idx(self, remaining_mask: np.ndarray, use_cache: bool = True) -> int:
        """Find best guess index using entropy, considering only remaining words."""
        # Check cache first (before np.where for speed)
        if use_cache:
            cache_key = remaining_mask.tobytes()
            cached = self._best_guess_cache.get(cache_key)
            if cached is not None:
                return cached

        remaining_indices = np.where(remaining_mask)[0]
        n_remaining = len(remaining_indices)

        if n_remaining == 1:
            result = remaining_indices[0]
            if use_cache:
                self._best_guess_cache[cache_key] = result
            return result

        # Get frequencies for remaining words
        remaining_freqs = self.frequencies[remaining_indices].astype(np.float64)
        total_freq = remaining_freqs.sum()

        if total_freq == 0:
            result = remaining_indices[0]
            if use_cache:
                self._best_guess_cache[cache_key] = result
            return result

        # Get result submatrix: (n_remaining_guesses, n_remaining_answers)
        result_submatrix = self.result_matrix[remaining_indices][:, remaining_indices]

        # Use numba-optimized entropy computation
        best_local_idx = _find_best_guess(result_submatrix, remaining_freqs, total_freq)

        result = remaining_indices[best_local_idx]
        if use_cache:
            self._best_guess_cache[cache_key] = result
        return result

    def play_game(self, answer_idx: int, starting_idx: int, max_guesses: int = 6) -> int:
        """Play a game and return number of guesses (or max_guesses + 1 if failed)."""
        remaining_mask = np.ones(self.n_words, dtype=bool)
        guess_idx = starting_idx

        for turn in range(1, max_guesses + 1):
            if guess_idx == answer_idx:
                return turn

            result = self.result_matrix[guess_idx, answer_idx]
            remaining_mask = self.get_remaining_mask(guess_idx, result, remaining_mask)

            if not remaining_mask.any():
                return max_guesses + 1

            guess_idx = self.best_guess_idx(remaining_mask)

        if guess_idx == answer_idx:
            return max_guesses

        return max_guesses + 1

    def evaluate_starting_word(
        self,
        starting_word: str,
        max_guesses: int = 6,
        fail_penalty: float = 10.0,
        show_progress: bool = False,
    ) -> dict:
        """
        Evaluate a starting word against all answers.

        Uses caching of best_guess computations within the evaluation.
        """
        # Clear cache for new evaluation
        self._best_guess_cache.clear()

        starting_idx = self.word_to_idx[starting_word]

        distribution: dict[int, int] = {}
        total_score = 0.0
        failures = 0

        iterator = self.answer_indices
        if show_progress:
            iterator = tqdm(self.answer_indices, desc=f"Eval {starting_word}")

        for answer_idx in iterator:
            guesses = self.play_game(answer_idx, starting_idx, max_guesses)

            if guesses > max_guesses:
                failures += 1
                total_score += fail_penalty
                distribution[max_guesses + 1] = distribution.get(max_guesses + 1, 0) + 1
            else:
                total_score += guesses
                distribution[guesses] = distribution.get(guesses, 0) + 1

        total_games = len(self.answer_indices)
        return {
            "starting_word": starting_word,
            "average_score": total_score / total_games if total_games > 0 else 0,
            "failure_rate": failures / total_games if total_games > 0 else 0,
            "distribution": dict(sorted(distribution.items())),
            "total_games": total_games,
        }

    def rank_all_starting_words(
        self,
        max_guesses: int = 6,
        fail_penalty: float = 10.0,
        n_workers: int | None = None,
        parallel: bool = True,
    ) -> list[dict]:
        """Rank ALL words as starting words.

        Args:
            max_guesses: Maximum guesses per game
            fail_penalty: Penalty for failed games
            n_workers: Number of parallel workers (default: CPU count)
            parallel: Use parallel processing (default: True)
        """
        if not parallel:
            # Sequential fallback
            results = []
            for i in tqdm(range(self.n_words), desc="Ranking all starting words"):
                starting_word = self.words[i]
                result = self.evaluate_starting_word(
                    starting_word, max_guesses, fail_penalty, show_progress=False
                )
                results.append(result)
            results.sort(key=lambda x: x["average_score"])
            return results

        # Parallel evaluation
        if n_workers is None:
            n_workers = cpu_count()

        # Prepare shared data for workers
        init_args = (
            self.words,
            self.frequencies,
            self.result_matrix,
            self.answer_indices,
            max_guesses,
            fail_penalty,
        )

        results = []
        with Pool(
            processes=n_workers,
            initializer=_init_eval_worker,
            initargs=init_args,
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_evaluate_starting_worker, range(self.n_words)),
                total=self.n_words,
                desc="Ranking all starting words (parallel)",
            ):
                results.append(result)

        results.sort(key=lambda x: x["average_score"])
        return results

    def rank_starting_words(
        self,
        candidates: list[str],
        max_guesses: int = 6,
        fail_penalty: float = 10.0,
    ) -> list[dict]:
        """Rank specific candidate starting words."""
        results = []

        for starting_word in tqdm(candidates, desc="Ranking candidates"):
            result = self.evaluate_starting_word(
                starting_word, max_guesses, fail_penalty, show_progress=False
            )
            results.append(result)

        results.sort(key=lambda x: x["average_score"])
        return results