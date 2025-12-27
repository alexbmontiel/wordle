from collections import OrderedDict
from itertools import islice

from wordfreq import get_frequency_dict


def create_word_list() -> OrderedDict:
    all_words = get_frequency_dict('en', "best")
    five_letter_words = {i.upper(): j for i, j in all_words.items() if len(i) == 5 and all(j.isalpha() for j in i)}
    five_letter_words = OrderedDict(reversed(sorted(five_letter_words.items(), key=lambda x: x[1])))

    return five_letter_words

def filter_words(word_list: OrderedDict, n: int) -> OrderedDict[str, float]:
    return OrderedDict(islice(word_list.items(), n))
