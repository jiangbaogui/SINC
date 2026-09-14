import unittest

import torch

import sinc
from onboard.core.onboard_inference import OnboardInference


class PublicApiTests(unittest.TestCase):
    def test_public_inference_alias_resolves_to_current_implementation(self):
        self.assertIs(sinc.SINCInference, OnboardInference)
        self.assertIs(sinc.OnboardInference, OnboardInference)

    def test_log_variance_is_converted_to_standard_deviation(self):
        log_variance = torch.tensor([0.0, 2.0])
        standard_deviation = OnboardInference._log_variance_to_standard_deviation(
            log_variance
        )
        torch.testing.assert_close(
            standard_deviation, torch.tensor([1.0, torch.e])
        )


if __name__ == "__main__":
    unittest.main()
