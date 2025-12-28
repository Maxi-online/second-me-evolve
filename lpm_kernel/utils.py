import logging
from enum import Enum
import tiktoken
import re
from typing import Any, Optional, Union, Collection, AbstractSet, Literal, List
from langchain.text_splitter import TextSplitter
import random
import string
from itertools import chain
import json
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

class IntentType(Enum):
    Emotion = "Emotion"
    Knowledge = "Knowledge"


def select_language_desc(
    preferred_language,
    default_desc="Identify the language of the provided Hint. Your response must be in the same language.",
):
    custom_desc = "You must respond in {}."
    if isinstance(preferred_language, str) and "/" in preferred_language:
        native, es = preferred_language.split("/")
        logging.info(f"Native: {native}, ES: {es}")
        return custom_desc.format(es)
    else:
        logging.info(
            "Error: preferred_language is not in the correct format. It should be 'native/es'."
        )
        return default_desc


def cal_upperbound(
    model_limit: int = 4096,
    generage_limit: int = 512,
    tolerance: int = 500,
    raw: str = "",
    model_name: str = "gpt-3.5-turbo",
) -> int:
    """
    :param model_limit: Maximum token count for the underlying model call
    :param tolerance: Error tolerance buffer
    :param raw: system prompt and raw content
    :return:
    """
    if model_name is not None:
        if model_name in tiktoken.model.MODEL_TO_ENCODING:
            enc = tiktoken.encoding_for_model(model_name)
            logging.info(f"Successfully initialized tokenizer for model: {model_name}")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
            logging.warning(f"Model '{model_name}' doesn't have a corresponding tokenizer, falling back to default: cl100k_base")
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        logging.info(f"No model specified, using default tokenizer: cl100k_base")
    raw_token = len(enc.encode(raw))
    upper_bound = model_limit - raw_token - tolerance - generage_limit
    if upper_bound < 0:
        logging.info(f"raw content is too long: {raw_token}")
        return 0
    return upper_bound


def equidistant_filter(chunks, separator, filtered_chunks_n=6):
    # Select the first and last two chunks, sample the remaining chunks evenly from the middle
    gap = (len(chunks) - 2) / (filtered_chunks_n - 2)
    indexes = [
        int(gap * i)
        for i in range(int(len(chunks) / gap) + 1)
        if (gap * i < len(chunks) - 2)
    ]
    filtered_chunks = [chunks[i] for i in indexes]
    filtered_chunks.append(separator.join(chunks[-2:]))
    return filtered_chunks


def tab_or_space_replacement(match):
    # If there is a tab character in the matched string, replace it with a single tab, otherwise replace it with a single space
    return "\t" if "\t" in match.group() else " "


def text_filter(text: str) -> str:
    pattern_tab_space = "[ \t]{3,}"
    pattern_wordwrap = "[\n\f\r\v]{3,}"
    # Replace when encountering three or more spaces or tabs
    replaced_text = re.sub(pattern_tab_space, tab_or_space_replacement, text)
    # When there are multiple consecutive \n (newline), \f (form feed), \r (carriage return), \v (vertical tab), replace them with 2 original newlines
    replaced_text = re.sub(pattern_wordwrap, "\n\n", replaced_text)
    return replaced_text


ALLOW_SPECIAL_TOKEN = {"<|endofprompt|>", "<|endoftext|>"}


def find_sublist_indices(main_list, sublist):
    indices = []
    length = len(sublist)
    for i in range(len(main_list) - length + 1):
        if main_list[i : i + length] == sublist:
            indices.append((i, i + length))
    return indices


class TokenTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at tokens."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = ALLOW_SPECIAL_TOKEN,
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3 encoder instance
        if model_name is not None:
            if model_name in tiktoken.model.MODEL_TO_ENCODING:
                enc = tiktoken.encoding_for_model(model_name)
                logging.info(f"Successfully initialized tokenizer for model: {model_name}")
            else:
                enc = tiktoken.get_encoding(encoding_name)
                logging.warning(f"Model '{model_name}' doesn't have a corresponding tokenizer, falling back to default: {encoding_name}")
        else:
            enc = tiktoken.get_encoding(encoding_name)
            logging.info(f"No model specified, using default tokenizer: {encoding_name}")
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # Filter content with a large number of whitespace characters in the input text to increase the proportion of effective content within chunks
        text = text_filter(text)
        splits = []
        input_ids = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

        start_idx = 0
        while start_idx < len(input_ids):
            cur_idx = min(start_idx + self._chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
            s = self._tokenizer.decode(chunk_ids).strip()
            if s:
                s = self._cut_meaningless_head_tail(s)
                if s:
                    splits.append(s)
            start_idx += self._chunk_size - self._chunk_overlap
        logging.debug("finished split_text(): %s splits", len(splits))
        return splits

    def _cut_meaningless_head_tail(self, text: str) -> str:
        # Only split when there are multiple newlines, as parsing of PDF/Word often contains false newlines
        sentences = re.split("\. |! |\? |。|！|？|\n+ *\n+", text)
        if len(sentences) < 2:
            return text
        head = sentences[0]
        body = ". ".join(sentences[1:-1])
        tail = sentences[-1]
        head_len = len(
            self._tokenizer.encode(
                body,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
        )
        body_len = len(
            self._tokenizer.encode(
                body,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
        )
        tail_len = len(
            self._tokenizer.encode(
                tail,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
        )
        parts = []
        # Use length to roughly estimate the impact of discarding the tail; if the impact is not significant, discard it
        # Rough estimate: Chinese 20 tokens, 8 characters; English 10 tokens, 30 characters
        if head_len >= 20 or len(head) >= 30:
            parts.append(head)
        if body_len > 0:
            parts.append(body)
        if tail_len >= 20 or len(tail) >= 30:
            parts.append(tail)
        res = "\n".join(parts)

        logger.info(
            "_cut_meaningless_tail() removes redundant sentence tails from chunks, before cut: %s characters, after cut: %s characters",
            len(text),
            len(res),
        )
        return res


def chunk_filter(
    chunks, filter, filtered_chunks_n=6, separator="\n", spacer="\n……\n……\n……\n"
):
    if len(chunks) <= filtered_chunks_n:
        return separator.join(chunks)
    return spacer.join(filter(chunks, separator, filtered_chunks_n))


def get_safe_content_turncate(content, model_name="gpt-3.5-turbo", max_tokens=3300):
    if model_name is not None:
        if model_name in tiktoken.model.MODEL_TO_ENCODING:
            enc = tiktoken.encoding_for_model(model_name)
            logging.info(f"Successfully initialized tokenizer for model: {model_name}")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
            logging.warning(f"Model '{model_name}' doesn't have a corresponding tokenizer, falling back to default: cl100k_base")
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        logging.info(f"No model specified, using default tokenizer: cl100k_base")
    logging.warning(
        "get_safe_content_turncate(): current model maximum input length is %s, current input length is %s",
        max_tokens,
        len(enc.encode(content)),
    )
    if len(enc.encode(content)) > max_tokens:
        content = enc.decode(enc.encode(content)[:max_tokens])
    return content


class DataType(Enum):
    DOCUMENT = "DOCUMENT"
    WEBSITE = "WEBSITE"
    IMAGE = "IMAGE"
    TABLE = "TABLE"
    AUDIO = "AUDIO"
    TEXT = "TEXT"

    @staticmethod
    def extra_values_map():
        return {
            "SHORT_AUDIO": "AUDIO",
        }

    @classmethod
    def _missing_(cls, value):
        # Try to find the corresponding primary key value from the extra value mapping
        extra_map = cls.extra_values_map()
        if value in extra_map:
            value = extra_map[value]
            return cls.__members__.get(value)
        # If not found, return DOCUMENT by default
        logging.error("DataType._missing_(): Could not find corresponding DataType enum value: %s", value)
        return cls.DOCUMENT


def get_urls(string):
    url_arr = []

    if not string:
        return url_arr

    pattern = re.compile(
        r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;\u4e00-\u9fa5]+[-A-Za-z0-9+&@#/%=~_|]"
    )
    matcher = pattern.finditer(string)

    for match in matcher:
        url_arr.append(match.group())

    sorted_url_arr = sorted(set(url_arr), key=len, reverse=True)

    return sorted_url_arr


def get_random_string(s_length: int) -> str:
    # Generate a random string
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(s_length))


def get_random_strings(n: int, s_length: int) -> List[str]:
    unique_strings = set()
    while len(unique_strings) < n:
        unique_strings.add(get_random_string(s_length))
    return list(unique_strings)


def encode_urls(text, random_string_len: int = 16):
    urls = get_urls(text)
    random_strings = get_random_strings(len(urls), random_string_len)
    url2string_dict = dict(zip(urls, random_strings))
    string2url_dict = dict(zip(random_strings, urls))
    for url, random_string in url2string_dict.items():
        text = text.replace(url, random_string)
    return text, string2url_dict


def decode_urls(text, string2url_dict):
    for random_string, url in string2url_dict.items():
        text = text.replace(random_string, url)
    return text


class TokenParagraphSplitter(TextSplitter):
    """For business data characteristics, perform some additional processing. This includes:
    1. Complete fragments as independent chunks help improve information focus in each chunk. Complete fragments are mainly determined by period+newline.
    2. When complete fragments are too long, split them into sentences and combine sentences into chunks that meet window size limits
    3. If a sentence is too long, split it directly by token granularity
    """

    line_break_characters = ["\n", "\f", "\r", "\v"]
    whitespace_characters = [" ", "\t"]
    sentence_terminators = [
        ".",
        "!",
        "?",
        "。",
        "！",
        "？",
        "……",
        "...",
    ] + line_break_characters
    paired_punctuation = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
        ("<", ">"),
        ("“", "”"),
        ("‘", "’"),
        ("《", "》"),
        ("【", "】"),
    ]
    intra_sentence_delimiters = [",", "，", ";", "；"] + whitespace_characters

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        allowed_special: Union[Literal["all"], AbstractSet[str]] = ALLOW_SPECIAL_TOKEN,
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3 encoder instance
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        chunks = []

        # Clean up abnormal whitespace characters in the text, such as replacing 3 or more consecutive \n with \n\n
        text = text_filter(text)

        # Replace URLs in the text to avoid symbols like ./?/ in URLs interfering with sentence splitting
        text, string2url_dict = encode_urls(text)
        url_strings = list(string2url_dict.keys())

        # Split by paragraphs according to rules
        paragraphs = self._split_to_paragraphs(
            text, min_paragraph_length=self._chunk_size // 2
        )

        for i, paragraph in enumerate(paragraphs):
            splits = self._split_to_chunks(paragraph, url_strings)
            logging.debug(
                "paragraph %s/%s %s characters: %s",
                i + 1,
                len(paragraphs),
                len(paragraph),
                paragraph,
            )
            logging.debug(
                "paragraph %s/%s split into %s chunks: %s",
                i + 1,
                len(paragraphs),
                len(splits),
                splits,
            )
            chunks.extend(splits)

        chunks = [decode_urls(chunk, string2url_dict) for chunk in chunks]

        return chunks

    def _split_to_chunks(self, text: str, url_strings: List[str] = []) -> List[str]:
        sentences = self._split_to_sentences(text, url_strings)
        chunks = self._merge_sentences_into_chunks(
            sentences, min_chunk_size=self._chunk_size // 2
        )
        return chunks

    def _split_to_paragraphs(
        self, text: str, min_paragraph_length: int = 0
    ) -> List[str]:
        """Currently split the original document into paragraphs directly based on the \n[any space]\n rule."""
        line_break_characters = "".join(self.line_break_characters)
        whitespace_characters = "".join(self.whitespace_characters)
        paragraphs = re.split(
            f"([{line_break_characters}]+[{whitespace_characters}]*[{line_break_characters}])+",
            text,
        )
        if len(paragraphs) % 2 == 1:
            paragraphs = [""] + paragraphs
        paragraphs = [
            (paragraphs[i], paragraphs[i + 1])
            for i in range(0, len(paragraphs), 2)
            if (paragraphs[i] + paragraphs[i + 1]).strip()
        ]

        if not paragraphs:
            return []

        new_paragraphs = []
        cur_paragraph, cur_paragraph_len = "", 0

        # merge short or broken paragraphs
        for sep, paragraph in paragraphs:
            if cur_paragraph_len >= min_paragraph_length and any(
                cur_paragraph.endswith(sym) for sym in self.sentence_terminators
            ):
                new_paragraphs.append(cur_paragraph.strip())
                cur_paragraph, cur_paragraph_len = "", 0

            cur_paragraph_len += len(self._tokenizer.encode(sep + paragraph))
            cur_paragraph += sep + paragraph

        if cur_paragraph:
            new_paragraphs.append(cur_paragraph.strip())

        return new_paragraphs

    def _split_to_sentences(self, text: str, url_strings: List[str] = []) -> List[str]:
        # Use capture groups to preserve sentence separators
        pattern = (
            f"({'|'.join(re.escape(symbol) for symbol in self.sentence_terminators)})+"
        )
        parts = re.split(pattern, text)
        sentences = []
        # Merge by skipping steps to ensure punctuation is added to the end of the corresponding sentence
        if len(parts) % 2 == 1:
            parts.append("")

        sentences = ["".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]

        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return []

        # Fix fragmented sentences, mainly for special cases such as numeric indices, floating-point numbers, etc., which may be separated
        sentences = self.recombine_broken_sentences(sentences)

        # Split sentences that are too long; in the short term, split directly by character length; future optimizations could consider splitting by punctuation within sentences
        sentences_list = [
            self._force_split_to_chunks(s, url_strings) for s in sentences
        ]
        sentences = list(chain.from_iterable(sentences_list))
        return sentences

    def recombine_broken_sentences(self, sentences: List[str]) -> List[str]:
        """Fix fragmented sentences, mainly for special cases such as numeric indices, floating-point numbers, etc., which may be separated。"""
        if len(sentences) < 2:
            return sentences

        open_symbols_dict = {
            open_sym: close_sym for open_sym, close_sym in self.paired_punctuation
        }
        close_symbols_dict = {
            close_sym: open_sym for open_sym, close_sym in self.paired_punctuation
        }

        new_sentences = []
        cur_sentences = ""
        unmatched_symbol = []

        for sent in sentences:
            # If the current sentence is not empty, doesn't meet predefined merge conditions, and has no pending matching punctuation ([, (, {, etc.), then consider the sentence complete
            if cur_sentences.strip() and not (
                self.check_merge(cur_sentences, sent) or unmatched_symbol
            ):
                new_sentences.append(cur_sentences)
                cur_sentences = ""

            for c in sent:
                if c in open_symbols_dict:
                    unmatched_symbol.append(c)
                elif c in close_symbols_dict:
                    if (
                        unmatched_symbol
                        and unmatched_symbol[-1] == close_symbols_dict[c]
                    ):
                        unmatched_symbol.pop()

                # By default, the current sentence ends when a newline-like character appears
                if c in self.line_break_characters:
                    unmatched_symbol = []
                    if cur_sentences.strip():
                        new_sentences.append(cur_sentences)
                        cur_sentences = ""
                cur_sentences += c

        if cur_sentences:
            new_sentences.append(cur_sentences)

        return new_sentences

    def check_merge(self, pre_sen, cur_sen):
        if len(pre_sen) > 1 and len(cur_sen) > 0:
            # If it's a decimal point in the middle of a floating-point number
            if pre_sen[-1] == "." and pre_sen[-2].isdigit() and cur_sen[0].isdigit():
                return True
            # If it's a numeric index at the beginning of a sentence, such as 1. *****\n2. *****
            if (
                pre_sen[-1] == "."
                and pre_sen[-2].isdigit()
                and cur_sen[0] not in self.line_break_characters
            ):
                return True
            # In markdown format, ! followed by [ may be an image link
            if (
                pre_sen[-1] == "!"
                and pre_sen[-2] in self.line_break_characters
                and cur_sen[0] == "["
            ):
                return True

        return False

    def _merge_sentences_into_chunks(
        self, sentences: List[str], min_chunk_size: int = 200
    ) -> List[str]:
        """Assemble into chunks according to chunk_size and overlap. Note that external guarantees ensure that the length of a single sentence does not exceed chunk_size"""
        if not sentences:
            return []

        n_tokens = [
            len(
                self._tokenizer.encode(
                    sentence,
                    allowed_special=self._allowed_special,
                    disallowed_special=self._disallowed_special,
                )
            )
            for sentence in sentences
        ]

        chunks = []
        start_idx = 0
        end_idx = start_idx + 1
        cur_token_num = n_tokens[start_idx]
        while start_idx < len(n_tokens):
            # Tail reaches the end point,
            if end_idx >= len(n_tokens):
                chunk = "".join(sentences[start_idx:end_idx])
                logging.debug(
                    "sentences[%s:%s] merged into chunk, current num_tokens: %s(%s)",
                    start_idx,
                    end_idx,
                    sum(n_tokens[start_idx:end_idx]),
                    cur_token_num,
                )
                chunks.append(chunk)
                break
            else:
                # +The next sentence will not exceed chunk_size, continue to include new sentences
                if cur_token_num + n_tokens[end_idx] <= self._chunk_size:
                    cur_token_num += n_tokens[end_idx]
                    end_idx += 1
                # +The next sentence will exceed chunk_size, assemble the current chunk and move to the next chunk
                else:
                    chunk = "".join(sentences[start_idx:end_idx])
                    logging.debug(
                        "sentences[%s:%s] merged into chunk, current num_tokens: %s(%s)",
                        start_idx,
                        end_idx,
                        sum(n_tokens[start_idx:end_idx]),
                        cur_token_num,
                    )
                    chunks.append(chunk)
                    # Next chunk: idx moves at least one position forward, start_idx allows overlap
                    end_idx = end_idx + 1
                    # Find a new starting point for start_idx that doesn't exceed the overlap
                    new_start_idx = end_idx - 1
                    overlap = 0
                    new_cur_token_num = n_tokens[new_start_idx]
                    while new_start_idx > start_idx + 1:
                        if (
                            overlap + n_tokens[new_start_idx - 1] >= self._chunk_overlap
                            or new_cur_token_num >= self._chunk_size
                        ):
                            break
                        new_start_idx -= 1
                        overlap += n_tokens[new_start_idx]
                        new_cur_token_num += n_tokens[new_start_idx]

                    start_idx = new_start_idx
                    cur_token_num = new_cur_token_num
        if len(chunks) > 1 and len(chunks[-1]) < min_chunk_size:
            logging.warning(
                "The last chunk length %s is less than %s, merge with the previous chunk",
                len(chunks[-1]),
                min_chunk_size,
            )
            last_chunk = chunks.pop()
            chunks[-1] += last_chunk

        chunks = [chunk for chunk in chunks if chunk.strip()]

        return chunks

    def _force_split_to_chunks(
        self, text: str, url_strings: List[str] = []
    ) -> List[str]:
        # TODO: In the future, consider adding forced splitting logic, such as: if a single sentence is too long, split by punctuation within the sentence, trying to preserve links and other data that require complete information
        """If a single sentence is too long, it can only be forcibly split, split by punctuation within the sentence, trying to preserve links and other data that require complete information"""
        splits = []
        input_ids = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
        if len(input_ids) < self._chunk_size:
            return [text]

        if text[-1] not in self.sentence_terminators + self.intra_sentence_delimiters:
            text += self.sentence_terminators[0]

        cur_sentence, cur_sentence_len = "", 0
        sub_sentence = ""
        for c in text:
            sub_sentence += c
            if c in self.intra_sentence_delimiters + self.sentence_terminators:
                sub_sentence_len = len(self._tokenizer.encode(sub_sentence))
                if (
                    cur_sentence_len + sub_sentence_len
                    > self._chunk_size - self._chunk_overlap
                ):
                    if cur_sentence:
                        splits.append(cur_sentence)
                        cur_sentence, cur_sentence_len = sub_sentence, sub_sentence_len
                    else:
                        # This indicates that sub_sentence is too long, at this point directly follow the forced splitting logic based on tokens
                        _splits = self.safe_split(sub_sentence, url_strings)
                        splits.extend(_splits[:-1])
                        cur_sentence, cur_sentence_len = _splits[-1], len(_splits[-1])
                else:
                    cur_sentence += sub_sentence
                    cur_sentence_len += sub_sentence_len
                sub_sentence = ""

        if cur_sentence:
            splits.append(cur_sentence)

        return splits

    def safe_split(self, sub_sentence: str, url_strings: List[str] = []) -> List[str]:
        sub_sentence_tokens = self._tokenizer.encode(sub_sentence)

        # Find the position intervals of all strings in url_strings
        url_string_intervals = []
        for url_string in url_strings:
            encoded_url_string = self._tokenizer.encode(url_string)
            # Use find_sublist_indices to find all position intervals
            url_string_intervals.extend(
                find_sublist_indices(sub_sentence_tokens, encoded_url_string)
            )

        _splits = []
        i = 0
        while i < len(sub_sentence_tokens):
            if i + self._chunk_size >= len(sub_sentence_tokens):
                slice_end = len(sub_sentence_tokens)
            else:
                slice_end = i + self._chunk_size - self._chunk_overlap

            # Determine if the split interval overlaps with any important string intervals
            for s_begin, s_end in url_string_intervals:
                if i < s_end <= slice_end or i < s_begin < slice_end:
                    slice_end = max(slice_end, s_end)

            # Split and record the current chunk
            _splits.append(self._tokenizer.decode(sub_sentence_tokens[i:slice_end]))
            # Move to the starting point of the next chunk
            i = slice_end

        return _splits


def get_summarize_title_keywords(responses):
    # Clean LLM generated content to obtain summarized text titles, abstracts, and keywords
    pattern = re.compile(r"\{.*(\}|\]|\,)", re.DOTALL)
    gen_texts = [each.choices[0].message.content for each in responses]
    logging.info("gen_texts: %s", gen_texts)
    results = []
    for res in gen_texts:
        try:
            # Match against the pattern
            matches = list(pattern.finditer(res))
            if not matches:
                results.append(("", "", []))
            else:
                answer = matches[0].group(0)
                content = answer.strip().strip(",")
                content += "]" * (content.count("[") - content.count("]"))
                content += "}" * (content.count("{") - content.count("}"))
                d = json.loads(res)
                results.append(
                    (d.get("title", ""), d.get("summary", ""), d.get("keywords", []))
                )

        except json.JSONDecodeError:
            logging.warning("JSON parsing failed, returning empty list")
            results.append(("", "", []))
    return results
