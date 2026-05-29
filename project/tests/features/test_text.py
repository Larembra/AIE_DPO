from features.text import tokenize, split_comma_words, comma_words_to_token_lists


def test_tokenize_basic():
    assert tokenize("Hello, world!") == ["hello", ",", "world", "!"]


def test_split_comma_words_and_comma_to_tokens():
    assert split_comma_words(None) == []
    assert split_comma_words("a, b, ,c") == ["a", "b", "c"]
    assert comma_words_to_token_lists("Hello, world") == [["hello"], ["world"]]

