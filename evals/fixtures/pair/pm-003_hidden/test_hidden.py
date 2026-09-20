"""Hidden judge: every listed module gains a distinct describe().

Generated, and generated per case, so the module list cannot drift from the
case that names it. It lives outside the fixture trees because those trees are
copied into the agent's sandbox: a test inside one is readable, and a case whose
answer is readable measures reading.
"""

import importlib

MODULES = ['socket', 'framing', 'backoff']


def test_each_module_exposes_a_non_empty_describe():
    for name in MODULES:
        module = importlib.import_module(f"modules.{name}")
        assert hasattr(module, "describe"), f"modules/{name}.py has no describe()"
        described = module.describe()
        assert isinstance(described, str), f"{name}.describe() is not a string"
        assert described.strip(), f"{name}.describe() returned nothing"


def test_describe_is_module_specific():
    """A constant passes the test above while doing none of the work.

    Without this second assertion the cheapest passing solution is to paste one
    string into all three modules, and the case would report a success that no
    reader of the diff would call one.
    """
    texts = {
        importlib.import_module(f"modules.{name}").describe()
        for name in MODULES
    }
    assert len(texts) == len(MODULES), "describe() must differ per module"
