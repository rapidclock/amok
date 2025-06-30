"""Tests for the utils module."""

from amok.utils import surround_with_tags


def test_surround_with_tags():
    """Test the surround_with_tags function."""
    assert surround_with_tags("foo", "BAR") == "<BAR>foo</BAR>"


def test_surround_with_tags_empty_content():
    """Test the surround_with_tags function with empty content."""
    assert surround_with_tags("", "BAR") == "<BAR></BAR>"
