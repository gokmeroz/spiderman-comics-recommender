"""
test_cli_smoke.py

Smoke test for the CLI demo setup (Version 6).

We don't test interactive input() flows — those require mocking stdin
and add complexity for little gain. Instead we verify that:
  1. build_recommenders() initializes without errors.
  2. All five mode handler functions can be imported.
"""

from app.demo import build_recommenders


def test_build_recommenders_runs_without_error():
    """All recommenders should initialize cleanly from the real data files."""
    items, interactions, content_rec, collab_rec, profile_rec, hybrid_rec = (
        build_recommenders()
    )
    assert items is not None
    assert interactions is not None
    assert content_rec is not None
    assert collab_rec is not None
    assert profile_rec is not None
    assert hybrid_rec is not None


def test_demo_modes_are_importable():
    """All mode handler functions must be importable from app.demo."""
    from app.demo import (
        mode_query,
        mode_profile,
        mode_synthetic_user,
        mode_hybrid,
        mode_evaluation,
    )
    assert callable(mode_query)
    assert callable(mode_profile)
    assert callable(mode_synthetic_user)
    assert callable(mode_hybrid)
    assert callable(mode_evaluation)
