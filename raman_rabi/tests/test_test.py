from unittest import TestCase

from raman_rabi import test

class TestTest(TestCase):
    def test_is_string(self):
        s = test.test()
        self.assertTrue(isinstance(s, str))
