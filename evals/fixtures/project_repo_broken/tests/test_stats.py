def test_mean_of_three():
    from projectkit.stats import mean

    assert mean([1, 2, 3]) == 2


def test_mean_of_single():
    from projectkit.stats import mean

    assert mean([4]) == 4
