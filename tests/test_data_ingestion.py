import unittest
import os
from src.components.data_ingestion import create_dataloader
from src.config.configuration import ConfigurationManager

class TestDataIngestion(unittest.TestCase):
    
    def setUp(self):
        """ Load configuration before each test """
        self.config_manager = ConfigurationManager()
        self.data_config = self.config_manager.get_data_ingestion_config()
    
    def test_data_directory_exists(self):
        """ Test if the data directory exists after ingestion """
        create_dataloader(self.data_config.root_dir, self.data_config.batch_size)
        self.assertTrue(os.path.exists(self.data_config.root_dir))
    
if __name__ == "__main__":
    unittest.main()
