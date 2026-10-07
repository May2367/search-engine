from engine.tokenizer import tokenizer


def test_lowercases_input():
    assert tokenizer("Python PYTHON python") == [
        "python",
        "python",
        "python",
    ]


def test_removes_punctuation():
    assert tokenizer("hello, world!") == [
        "hello",
        "world",
    ]


def test_handles_multiple_spaces():
    assert tokenizer("hello   world") == [
        "hello",
        "world",
    ]


def test_handles_empty_document():
    assert tokenizer("") == []


def test_handles_only_punctuation():
    assert tokenizer("!!!...,,,") == []


def test_removes_non_letters():
    assert tokenizer("python 123 testing") == [
        "python",
        "testing",
    ]


def test_preserves_duplicate_words():
    assert tokenizer("python python java") == [
        "python",
        "python",
        "java",
    ]
