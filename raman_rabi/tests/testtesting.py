from unittest import TestCase

import raman_rabi
from raman_rabi import testing

class TestTesting(TestCase):
    def test_is_string(self):
        s = testing.hello()
        self.assertTrue(isinstance(s, str))
