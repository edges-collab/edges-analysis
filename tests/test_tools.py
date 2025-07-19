from edges import tools


def test_dct_to_list():
    """Ensure simple dictionary is dealt with correctly."""
    dct_of_lists = {"a": [1, 2], "b": [3, 4]}

    list_of_dicts = tools.dct_of_list_to_list_of_dct(dct_of_lists)

    assert list_of_dicts == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
