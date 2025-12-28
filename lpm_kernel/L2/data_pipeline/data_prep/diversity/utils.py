from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Any
import logging


def string_similarity(str1: str, str2: str) -> float:
    """Calculate the edit distance similarity between two strings.

    Args:
        str1: First string for comparison.
        str2: Second string for comparison.

    Returns:
        float: Similarity ratio between 0.0 and 1.0.
    """
    return SequenceMatcher(None, str1, str2).ratio()


def remove_similar_dicts(dict_list: List[Dict[str, Any]], similarity_threshold: float = 0.6) -> Tuple[List[Dict[str, Any]], int]:
    """Remove dictionaries with content field similarity greater than threshold.

    Args:
        dict_list: List of dictionaries containing 'content' field to check for similarity.
        similarity_threshold: Maximum similarity allowed between items (default: 0.6).

    Returns:
        Tuple containing:
            - List of dictionaries after removing similar items.
            - Count of similar items found.
    """
    unique_dicts = []
    cnt = 0
    for i, current_dict in enumerate(dict_list):
        if not current_dict["content"]:
            continue
        is_similar = False
        for j in range(len(unique_dicts)):
            if not unique_dicts[j]["content"]:
                continue
            if (
                string_similarity(current_dict["content"], unique_dicts[j]["content"])
                > similarity_threshold
            ):
                is_similar = True
                logging.info(
                    f" {current_dict['content'][-100:]}\n is similar to: \n{unique_dicts[j]['content'][-100:]}\n____________________"
                )
                cnt += 1
                break

        if not is_similar:
            unique_dicts.append(current_dict)

    return unique_dicts, cnt