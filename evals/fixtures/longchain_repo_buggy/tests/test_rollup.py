def test_total_of_values():
    from rollup import total

    assert total([1, 2, 3]) == 6


def test_total_of_empty():
    from rollup import total

    assert total([]) == 0
