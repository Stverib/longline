def test_slugify_simple():
    from projectkit.text import slugify

    assert slugify("Hello World") == "hello-world"
