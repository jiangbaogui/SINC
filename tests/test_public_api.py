import unittest

import sinc
from onboard.core.onboard_inference import OnboardInference


class PublicApiTests(unittest.TestCase):
    def test_public_inference_alias_resolves_to_current_implementation(self):
        self.assertIs(sinc.SINCInference, OnboardInference)
        self.assertIs(sinc.OnboardInference, OnboardInference)


if __name__ == "__main__":
    unittest.main()
