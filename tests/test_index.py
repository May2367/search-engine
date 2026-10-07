from engine.index import build_index


def test_builds_basic_index():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

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

    index = build_index(documents)

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

    index = build_index(documents)

    assert index == {
        "python": {
            1: 1,
        },
        "java": {
            3: 1,
        },
    }


def test_handles_empty_input():
    assert build_index([]) == {}


def test_normalizes_documents_before_indexing():
    documents = [
        "Python PYTHON python!",
        "JAVA java",
    ]

    index = build_index(documents)

    assert index == {
        "python": {
            0: 3,
        },
        "java": {
            1: 2,
        },
    }
