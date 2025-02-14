import unittest
from src.pipeline.stage_04_evaluation import EvaluationPipeline

class TestEvaluation(unittest.TestCase):
    
    def setUp(self):
        """ Initialize evaluation pipeline before each test """
        self.evaluation_pipeline = EvaluationPipeline()
    
    def test_evaluation_execution(self):
        """ Test if evaluation executes without errors """
        try:
            self.evaluation_pipeline.main()
            success = True
        except Exception as e:
            success = False
            print(f"Evaluation failed: {e}")
        self.assertTrue(success)
    
if __name__ == "__main__":
    unittest.main()
