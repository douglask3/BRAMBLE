"""Minimal "will it run" test"""

from bramble import Bramble

def test_init():
    b = Bramble({})
    assert isinstance(b, Bramble)
