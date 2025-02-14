import os
from pathlib import Path
import yaml
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig, TrainingConfig, EvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        create_directories([config["root_dir"]])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_URL=config["source_URL"],
            local_data_file=config["local_data_file"],
            unzip_dir=config["unzip_dir"]
        )
        return data_ingestion_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config["training"]
        params = self.params
        training_data = os.path.join(self.config["data_ingestion"]["unzip_dir"], "Car-Park-Images")
        create_directories([Path(training["root_dir"])])

        training_config = TrainingConfig(
            root_dir=Path(training["root_dir"]),
            trained_model_path=Path(training["trained_model_path"]),
            training_data=Path(training_data),
            params_epochs=params["EPOCHS"],
            params_batch_size=params["BATCH_SIZE"],
            params_learning_rate=params["LEARNING_RATE"]
        )
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/model.pth"),
            training_data=Path("artifacts/data_ingestion/Car-Park-Images"),
            all_params=self.params,
            params_batch_size=self.params["BATCH_SIZE"]
        )
        return eval_config