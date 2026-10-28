"""readout gate. Re-exports TestReadout from the canonical gate at
tests/test_e2_spt.py."""
from tests.test_spt_measurement_cost import TestReadout  # noqa: F401

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
