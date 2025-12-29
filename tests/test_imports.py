def test_imports():
    # Import your real package, not "src"
    import lyric_karaoke  # noqa: F401

    # Optional: import key modules to catch broken relative imports early
    import lyric_karaoke.features  # noqa: F401
    import lyric_karaoke.segments  # noqa: F401
    import lyric_karaoke.inference  # noqa: F401
    import lyric_karaoke.karaoke_logic  # noqa: F401
