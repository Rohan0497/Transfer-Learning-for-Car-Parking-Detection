import unittest
import os
from src.pipeline.predict import InferencePipeline

class TestInference(unittest.TestCase):
    
    def setUp(self):
        """ Initialize inference pipeline before each test """
        self.test_image = "tests/sample_test_image.jpg"
        self.inference_pipeline = InferencePipeline(self.test_image)
    
    def test_inference_execution(self):
        """ Test if inference executes without errors """
        try:
            result = self.inference_pipeline.main()
            success = isinstance(result, dict) and "detections" in result
        except Exception as e:
            success = False
            print(f"Inference failed: {e}")
        self.assertTrue(success)
    
if __name__ == "__main__":
    unittest.main()
