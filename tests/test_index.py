from engine.index import build_index


def test_builds_basic_index():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)["index"]

    assert index == {
        "python": {
            0: 1,
            1: 1,
        },
        "java": {
            0: 1,
            2: 1,
        },
        "rust": {
            1: 1,
        },
    }


def test_tracks_term_frequency():
    documents = [
        "python python python java",
        "python java java",
    ]

    index = build_index(documents)["index"]

    assert index["python"] == {
        0: 3,
        1: 1,
    }

    assert index["java"] == {
        0: 1,
        1: 2,
    }


def test_handles_empty_documents():
    documents = [
        "",
        "python",
        "",
        "java",
    ]

    index = build_index(documents)["index"]

    assert index == {
        "python": {
            1: 1,
        },
        "java": {
            3: 1,
        },
    }


def test_handles_empty_input():
    result = build_index([])
    assert result["index"] == {}
    assert result["corpus_stats"]["num_docs"] == 0
    assert result["corpus_stats"]["doc_freq"] == {}
    assert result["corpus_stats"]["doc_lengths"] == {}
    assert result["corpus_stats"]["avg_doc_length"] == 0.0

def test_normalizes_documents_before_indexing():
    documents = [
        "Python PYTHON python!",
        "JAVA java",
    ]

    index = build_index(documents)["index"]

    assert index == {
        "python": {
            0: 3,
        },
        "java": {
            1: 2,
        },
    }

def test_tracks_corpus_statistics():
    documents = [
        "python java python",
        "java rust",
        "python",
    ]

    result = build_index(documents)
    stats = result["corpus_stats"]

    assert stats["num_docs"] == 3
    assert stats["doc_freq"] == {
        "python": 2,
        "java": 2,
        "rust": 1,
    }
    assert stats["doc_lengths"] == {
        0: 3,
        1: 2,
        2: 1,
    }
    assert stats["avg_doc_length"] == 2.0
