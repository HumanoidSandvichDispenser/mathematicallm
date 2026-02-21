"""
Tests for strategy.py - Strategy class.
"""

import pytest
from sympy import S
from zermelo.trees.strategy import Strategy


class TestStrategy:
    def test_init(self):
        s = Strategy({"iset1": "action1", "iset2": "action2"})
        assert s["iset1"] == "action1"
        assert s["iset2"] == "action2"

    def test_len(self):
        s = Strategy({"a": "1", "b": "2", "c": "3"})
        assert len(s) == 3

    def test_iter(self):
        s = Strategy({"a": "1", "b": "2"})
        assert set(iter(s)) == {"a", "b"}

    def test_repr(self):
        s = Strategy({"a": "1", "b": "2"})
        assert repr(s) == "Strategy(a: 1, b: 2)"

    def test_eq(self):
        s1 = Strategy({"a": "1", "b": "2"})
        s2 = Strategy({"a": "1", "b": "2"})
        s3 = Strategy({"a": "1", "b": "3"})
        assert s1 == s2
        assert s1 != s3

    def test_hash(self):
        s1 = Strategy({"a": "1", "b": "2"})
        s2 = Strategy({"a": "1", "b": "2"})
        s3 = Strategy({"a": "1", "b": "3"})
        assert hash(s1) == hash(s2)
        assert hash(s1) != hash(s3)

    def test_concat(self):
        s1 = Strategy({"iset1": "action1"})
        s2 = Strategy({"iset2": "action2"})
        combined = s1.concat(s2)
        assert combined["iset1"] == "action1"
        assert combined["iset2"] == "action2"

    def test_concat_overwrites_first_with_second(self):
        s1 = Strategy({"iset1": "action1", "iset2": "old"})
        s2 = Strategy({"iset2": "new"})
        combined = s1.concat(s2)
        assert combined["iset1"] == "action1"
        assert combined["iset2"] == "new"

    def test_concat_returns_new_strategy(self):
        s1 = Strategy({"iset1": "action1"})
        s2 = Strategy({"iset2": "action2"})
        combined = s1.concat(s2)
        assert combined is not s1
        assert combined is not s2

    def test_concat_empty(self):
        s1 = Strategy({})
        s2 = Strategy({"iset1": "action1"})
        combined = s1.concat(s2)
        assert combined["iset1"] == "action1"

    def test_concat_both_empty(self):
        s1 = Strategy({})
        s2 = Strategy({})
        combined = s1.concat(s2)
        assert len(combined) == 0
