from itertools import islice

from wordfreq import get_frequency_dict


def is_simple_plural_or_verb(word: str, word_freq: float, all_words: dict[str, float]) -> bool:
    """Check if word is a simple plural/verb form (ends in S with common base form)."""
    if not word.endswith('S'):
        return False

    # Check potential base forms
    base_s = word[:-1].lower()   # DOORS -> door, DENTS -> dent
    base_es = word[:-2].lower()  # BOXES -> box

    base_freq_s = all_words.get(base_s, 0)
    base_freq_es = all_words.get(base_es, 0) if word.endswith('ES') else 0

    # Use the more common base form
    base_freq = max(base_freq_s, base_freq_es)

    # Filter if base form is at least as common as the S-form
    # This catches DOORS (door > doors) but keeps GRASS (gras < grass)
    if base_freq >= word_freq:
        return True

    return False


def create_word_list() -> dict[str, tuple[float, frozenset[str]]]:
    """Create word list with pre-computed character sets for fast filtering.

    Returns dict mapping word -> (frequency, character_set)
    Filters out simple plurals/verb forms (S/ES endings where base form exists).
    """
    all_words = get_frequency_dict('en', "best")

    five_letter_words = {}
    for w, freq in all_words.items():
        if len(w) == 5 and w.isalpha():
            word = w.upper()
            if not is_simple_plural_or_verb(word, freq, all_words):
                five_letter_words[word] = (freq, frozenset(word))

    # Sort by frequency descending
    five_letter_words = dict(sorted(five_letter_words.items(), key=lambda x: x[1][0], reverse=True))

    return five_letter_words

def filter_words(word_list: dict, n: int) -> dict[str, tuple[float, frozenset[str]]]:
    return dict(islice(word_list.items(), n))
