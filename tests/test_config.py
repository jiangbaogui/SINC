import unittest

from ground.core.NDCConfig import NDCConfig


class ConfigTests(unittest.TestCase):
    def test_example_config_parses(self):
        config = NDCConfig("configs/vegetation_fire.yaml")
        self.assertEqual(config.n_bands, 7)
        self.assertEqual(config.n_harmonics, 3)
        self.assertEqual(config.n_levels, 12)
        self.assertEqual(config.train_loss_type, "NLL")
        self.assertEqual(config.temporal_buffer_days, 0)
        self.assertTrue(config.input_quality_screened)
        self.assertEqual(config.random_seed, 42)


if __name__ == "__main__":
    unittest.main()
