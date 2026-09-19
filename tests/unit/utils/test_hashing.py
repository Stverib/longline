"""The digest helper, and the fact that the harness's copy is the same function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from longline.utils.hashing import sha256_bytes, sha256_file

if TYPE_CHECKING:
    from pathlib import Path


def test_digest_is_stable_and_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"hello")
    first = sha256_file(path)
    assert first == sha256_file(path)
    assert first == sha256_bytes(b"hello")
    assert first == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_a_missing_file_digests_to_missing_rather_than_raising(tmp_path: Path) -> None:
    """An absent artifact is a fact about the run, not an error."""
    assert sha256_file(tmp_path / "nope") == "missing"


def test_the_eval_helper_is_this_function() -> None:
    """`faults.sha256_file` must not be a second implementation.

    A second implementation is a second set of semantics: the day one of them
    gains a normalisation the other does not, every comparison across the
    boundary silently compares different things.
    """
    from longline.eval.faults import sha256_file as eval_sha256_file

    assert eval_sha256_file is sha256_file
