import pytest


def cast_keys_to_string(d, changed_keys=dict()):
    nd = dict()
    for key in d.keys():
        if not isinstance(key, str):
            casted_key = str(key)
            changed_keys[casted_key] = key
        else:
            casted_key = key
        if isinstance(d[key], dict):
            nd[casted_key], changed_keys = cast_keys_to_string(d[key], changed_keys)
        else:
            nd[casted_key] = d[key]
    return nd, changed_keys


def cast_keys_back(d, changed_keys):
    nd = dict()
    for key in d.keys():
        if key in changed_keys:
            original_key = changed_keys[key]
        else:
            original_key = key
        if isinstance(d[key], dict):
            nd[original_key], changed_keys = cast_keys_back(d[key], changed_keys)
        else:
            nd[original_key] = d[key]
    return nd, changed_keys


@pytest.mark.parametrize("d",
    [
        {1: {2: 3, 4: 5}, 'a': 'b'},
        {'Hello': 'World', 'How': 0, 0: "zero", "heya": {"people": "are awesome", 1: "one"}},
        {'h': {1: 1, 2: 2}}
    ]
)
def test_cast_keys_to_string_and_back(d):
    # d = {1: {2: 3, 4: 5}, 'a': 'b'}

    casted, changed_keys = cast_keys_to_string(d)

    def assert_keys_are_strings(d):
        for k,v in d.items():        
            if isinstance(v, dict):
                assert_keys_are_strings(v)
            else:            
                assert isinstance(k, str)
    print(casted)

    assert_keys_are_strings(casted)

    original, _ = cast_keys_back(casted, changed_keys)

    print(original)
    assert original == d


if __name__ == "__main__":
    test = {'h': {1: 1, 2: 2}}
    # test = {1: {2: 3, 4: 5}, 'a': 'b'}

    test_string, changed_keys = cast_keys_to_string(test)
    print(f"Everything should be string: {test_string}")
    test_original = cast_keys_back(test_string, changed_keys)
    print(test)
    print(test_original[0])
    print(test == test_original[0])
