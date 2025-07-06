"""Tests for the utils module."""

import pytest

from amok.utils import surround_with_tags


def test_surround_with_tags():
    """Test the surround_with_tags function."""
    assert surround_with_tags("foo", "BAR") == "<BAR>\nfoo\n</BAR>"


def test_surround_with_tags_empty_content():
    """Test the surround_with_tags function with empty content."""
    assert surround_with_tags("", "BAR") == ""


def test_surround_with_tags_whitespace_content():
    """Test the surround_with_tags function with whitespace-only content."""
    assert surround_with_tags("   ", "BAR") == ""
    assert surround_with_tags("\n\t  \n", "BAR") == ""


def test_surround_with_tags_none_content():
    """Test the surround_with_tags function with None content."""
    assert surround_with_tags(None, "BAR") == ""


def test_surround_with_tags_multiline_content():
    """Test the surround_with_tags function with multiline content."""
    content = "line1\nline2\nline3"
    result = surround_with_tags(content, "TEST")
    assert result == "<TEST>\nline1\nline2\nline3\n</TEST>"


def test_surround_with_tags_content_with_spaces():
    """Test surround_with_tags function with leading/trailing spaces."""
    assert surround_with_tags("  foo  ", "BAR") == "<BAR>\nfoo\n</BAR>"


def test_surround_with_tags_lowercase_tag():
    """Test the surround_with_tags function with lowercase tag."""
    assert surround_with_tags("content", "lower") == "<LOWER>\ncontent\n</LOWER>"


def test_surround_with_tags_mixed_case_tag():
    """Test the surround_with_tags function with mixed case tag."""
    assert surround_with_tags("content", "MiXeD") == "<MIXED>\ncontent\n</MIXED>"


def test_surround_with_tags_empty_tag():
    """Test the surround_with_tags function with empty tag."""
    with pytest.raises(ValueError, match="Tag cannot be None or empty"):
        surround_with_tags("content", "")


def test_surround_with_tags_whitespace_tag():
    """Test the surround_with_tags function with whitespace-only tag."""
    with pytest.raises(ValueError, match="Tag cannot be None or empty"):
        surround_with_tags("content", "   ")


def test_surround_with_tags_none_tag():
    """Test the surround_with_tags function with None tag."""
    with pytest.raises(ValueError, match="Tag cannot be None or empty"):
        surround_with_tags("content", None)


def test_surround_with_tags_none_text():
    """Test surround_with_tags with None text."""
    assert surround_with_tags(None, "tag") == ""


def test_surround_with_tags_empty_text():
    """Test surround_with_tags with empty text."""
    assert surround_with_tags(" ", "tag") == ""
