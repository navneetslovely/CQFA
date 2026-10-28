"""physics gate. Re-exports TestPhysics / TestFunctional from the canonical
gate at tests/test_e2_spt.py."""
from tests.test_spt_measurement_cost import TestPhysics, TestFunctional  # noqa: F401

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
