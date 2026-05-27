import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), re.UNICODE)


def split_comma_words(text: str) -> list[str]:
    if text is None:
        return []
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return parts


def comma_words_to_token_lists(text: str) -> list[list[str]]:
    return [tokenize(part) for part in split_comma_words(text)]

