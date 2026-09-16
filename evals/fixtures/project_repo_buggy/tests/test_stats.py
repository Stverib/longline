def test_mean_of_three():
    from projectkit.stats import mean

    assert mean([1, 2, 3]) == 2


def test_mean_of_single():
    from projectkit.stats import mean

    assert mean([4]) == 4


def test_median_is_sorted_middle():
    from projectkit.stats import median

    assert median([1, 2, 3]) == 2


def test_median_of_even_length_averages_the_two_middle():
    from projectkit.stats import median

    assert median([1, 2, 3, 4]) == 2.5
