
from src.config.configuration import ConfigurationManager
from src.components.inference import run_inference
from src import logger

STAGE_NAME = "Inference Stage"

class InferencePipeline:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def main(self):
        config = ConfigurationManager()
        run_inference(self.image_path)

if __name__ == "__main__":
    try:
        test_image = "test.jpg"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = InferencePipeline(test_image)
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

def predict(image_path: str):
    """Runs prediction on a given image"""
    pipeline = InferencePipeline(image_path)
    pipeline.main()

if __name__ == "__main__":
    test_image = "test.jpg"
    predict(test_image)