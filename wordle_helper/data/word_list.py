"""WordList type and configuration utilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wordle_helper.data.frequency import FrequencyConfig

# Type alias for word lists: word -> (frequency, character_set)
WordList = dict[str, tuple[float, frozenset[str]]]


@dataclass
class WordListConfig:
    """
    Configuration for word list creation.
    
    Attributes:
        max_words: Maximum number of words to include (top N by frequency)
        frequency_config: Optional frequency transformation parameters
        source: Source identifier for tracking
    """
    max_words: int | None = None
    frequency_config: "FrequencyConfig | None" = None
    source: str = "wordfreq"


def create_word_list(config: WordListConfig) -> WordList:
    """
    Create a word list according to the configuration.
    
    Args:
        config: Configuration specifying word list parameters
        
    Returns:
        WordList dictionary
    """
    from wordle_helper.data.corpus import load_raw_word_list, filter_words
    
    # Load raw word list
    raw_word_list = load_raw_word_list()
    
    # Convert to WordList format (with character sets)
    word_list: WordList = {}
    for word, freq in raw_word_list.items():
        word_list[word] = (freq, frozenset(word))
    
    # Sort by frequency descending
    word_list = dict(sorted(word_list.items(), key=lambda x: x[1][0], reverse=True))
    
    # Apply max_words filter if specified
    if config.max_words is not None:
        word_list = filter_words(word_list, config.max_words)
    
    # Apply frequency transformation if specified
    if config.frequency_config is not None:
        word_list = apply_frequency_transform(word_list, config.frequency_config)
    
    return word_list


def apply_frequency_transform(word_list: WordList, config: "FrequencyConfig") -> WordList:
    """
    Apply frequency transformation to a word list.
    
    Args:
        word_list: Original word list
        config: Frequency transformation configuration
        
    Returns:
        New word list with transformed frequencies
    """
    # Import here to avoid circular dependency
    from wordle_helper.data.frequency import transform_word_list
    
    return transform_word_list(word_list, k=config.k, x0=config.x0)

