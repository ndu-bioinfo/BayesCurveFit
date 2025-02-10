import unittest
import numpy as np

class TestRandomState(unittest.TestCase):
    def test_random_state(self):
        expected_array = np.array([
            42, 3107752595, 1895908407, 3900362577, 3030691166,
            4081230161, 2732361568, 1361238961, 3961642104, 867618704
        ], dtype=np.uint32)
        
        # Create the RandomState object and get the state
        state = np.random.RandomState(seed=42).get_state()[1][:10]
        
        # Assert that the state matches the expected values
        np.testing.assert_array_equal(state, expected_array)

if __name__ == '__main__':
    unittest.main()