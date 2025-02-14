import unittest
import torch
from src.pipeline.stage_03_training import TrainingPipeline

class TestTraining(unittest.TestCase):
    
    def setUp(self):
        """ Initialize training pipeline before each test """
        self.training_pipeline = TrainingPipeline()
    
    def test_training_execution(self):
        """ Test if training executes without errors """
        try:
            self.training_pipeline.main()
            success = True
        except Exception as e:
            success = False
            print(f"Training failed: {e}")
        self.assertTrue(success)
    
if __name__ == "__main__":
    unittest.main()