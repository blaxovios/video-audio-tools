import logging
from typing import NoReturn, Union
import sys
import rtoml
from os import path, makedirs


class GeneralFunctions(object):
    
    def __init__(self) -> None:
        pass

    def setup_logging(self, debug_filename: str = 'debug') -> None:
        """Attach default Cloud Logging handler to the root logger."""
        root_logger = logging.getLogger()
        
        # Clear existing handlers
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Running locally
        logging.basicConfig(level=logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Ensure the log directory exists
        log_dir = 'logs'
        if not path.exists(log_dir):
            makedirs(log_dir)

        # Add file handler to log to a file
        file_handler = logging.FileHandler(f'logs/{debug_filename}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info("Local logging setup complete")
        
        # Print current handlers after setup
        print("Handlers after setup:", root_logger.handlers)
        
    def load_toml(self, toml_file_path: str) -> Union[dict, NoReturn]:
        """ Load a toml file and return it as a dictionary

        Args:
            toml_file_path (str): _description_

        Returns:
            Union[dict, NoReturn]: _description_
        """
        try:
            f = open(toml_file_path, 'r')
            toml_loaded_dict = rtoml.load(f.read())
            return toml_loaded_dict
        except Exception as e:
            logging.error(e)
            return sys.exit(1)