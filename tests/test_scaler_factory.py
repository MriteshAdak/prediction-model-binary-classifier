import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.scaler_factory import ScalerFactory, NoScaler

class TestScalerFactory(unittest.TestCase):
    def test_create_scaler_standard(self):
        scaler = ScalerFactory.create_scaler("standard")
        
        self.assertIsInstance(scaler, StandardScaler)
            
    def test_create_scaler_unknown_raises_error(self):
        with self.assertRaises(ValueError):
            ScalerFactory.create_scaler("unknown_scaler")
                
    def test_no_scaler_behavior(self):
        scaler = NoScaler()
        X = np.array([[1, 2], [3, 4]])
        
        self.assertEqual(scaler.fit(X), scaler)
        
        np.testing.assert_array_equal(scaler.transform(X), X)
        np.testing.assert_array_equal(scaler.fit_transform(X), X)

    def test_register_and_list_scalers(self):
        class CustomScaler:
            pass
        
        ScalerFactory.register_scaler("custom", CustomScaler, "Custom Description")
        
        self.assertIn("custom", ScalerFactory.list_available_scalers())
        self.assertEqual(ScalerFactory.get_description("custom"), "Custom Description")