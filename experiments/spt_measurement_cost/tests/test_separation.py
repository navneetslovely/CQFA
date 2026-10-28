"""separation gate. Re-exports TestSeparation from the canonical gate at
tests/test_e2_spt.py (coherent flat / shadow grows / known-string flat integrity
guard / coherent beats shadow)."""
from tests.test_spt_measurement_cost import TestSeparation  # noqa: F401

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
