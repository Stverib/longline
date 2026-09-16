def test_table():
    from projectkit.report import render_table

    assert render_table([[1, 2]]) == "1,2"
