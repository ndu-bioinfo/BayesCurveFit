import unittest
import numpy as np

class TestRandomState(unittest.TestCase):
    def test_random_state(self):
        expected_array = np.array([2147483648
        ], dtype=np.uint32)
        
        # Create the RandomState object and get the state
        state = np.random.RandomState().get_state()[1][:1]
        
        # Assert that the state matches the expected values
        np.testing.assert_array_equal(state, expected_array)


if __name__ == '__main__':
    unittest.main()