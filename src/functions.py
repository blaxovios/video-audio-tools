from os import getenv
import logging
from typing import NoReturn, Union
import sys
import rtoml
import os


class GeneralFunctions(object):
    def __init__(self) -> None:
        pass
    
    def local_logger(self, file_path: str='logs/debug.log'):
        """ Set up a local logger

        Args:
            file_path (str, optional): _description_. Defaults to 'logs/debug.log'.
        """
        local_logging = getenv(key="LOCAL_LOGGING", default=False)
        if local_logging:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(file_path),
                    logging.StreamHandler()
                ]
            )
        return
        
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