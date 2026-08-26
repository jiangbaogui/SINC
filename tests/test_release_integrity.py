import unittest

from sinc.preflight import verify_data_manifest, verify_source_manifest


class ReleaseIntegrityTests(unittest.TestCase):
    def test_synchronized_source_manifest(self):
        result = verify_source_manifest()
        self.assertTrue(result.passed, result.detail)

    def test_example_data_manifest(self):
        result = verify_data_manifest()
        self.assertTrue(result.passed, result.detail)


if __name__ == "__main__":
    unittest.main()
