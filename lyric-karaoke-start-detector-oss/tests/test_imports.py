"""Smoke test: imports should work."""

def test_imports():
    import src.features  # noqa: F401
    import src.segments  # noqa: F401
    import src.karaoke_logic  # noqa: F401
