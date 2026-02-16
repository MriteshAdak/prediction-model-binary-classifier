import unittest
import pandas as pd
import numpy as np
from src.preprocessor import Preprocessor
from src.config import AppConfig, FeatureConfig

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.app_config = AppConfig(
            target_column='is_claim',
            features=FeatureConfig(
                id_columns=['policy_id'],
                boolean_columns=['is_parking_camera'],
                float_columns=['length'],
                ordinal_columns={'ncap_rating': ['1', '2', '3', '4', '5']},
                categorical_columns=['transmission_type']
            )
        )
        self.processor = Preprocessor(self.app_config)
        
        self.raw_data = pd.DataFrame({
            'policy_id': ['123', '456'],
            'is_parking_camera': ['Yes', 'No'],
            'is_claim': [1, 0],
            'length': ['4000', '4200'],
            'ncap_rating': ['3', '5'],
            'transmission_type': ['Manual', 'Automatic']
        })

    def test_process_logic(self):
        X, y = self.processor.process(self.raw_data)

        self.assertNotIn('policy_id', X.columns)
        self.assertTrue(X['is_parking_camera'].iloc[0])  
        self.assertFalse(X['is_parking_camera'].iloc[1]) 
        self.assertEqual(X['length'].dtype, float)
        self.assertNotIn('transmission_type', X.columns)
        self.assertEqual(len(y), 2)
        self.assertTrue(isinstance(y, pd.Series))

    def test_process_handles_missing_ncap_rating_gracefully(self):
        raw_data = pd.DataFrame({
            'policy_id': ['a','b'],
            'is_parking_camera': ['Yes', 'No'],
            'is_claim': [1, 0],
            'length': [4000, 4200],
            'transmission_type': ['Manual', 'Automatic']
        })
        X, y = self.processor.process(raw_data)
        self.assertNotIn('ncap_rating', X.columns)
        self.assertEqual(len(y), 2)

    def test_process_raises_when_target_missing(self):
        bad_data = self.raw_data.drop(columns=['is_claim'])
        with self.assertRaises(ValueError):
            self.processor.process(bad_data)