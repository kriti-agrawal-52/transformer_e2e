import yaml
import torch
import logging
from types import SimpleNamespace
import os

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yml', as_namespace: bool=True):
    """
    Loads configuration from a yaml file.
    
    Args:
        config_path (str): Path to the YAML config file.
        as_namespace (bool): if true, returns configurations as SimpleNamespace type instead of dictionary
    
    Returns:
        SimpleNamespace or dict
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)  # loads the yml file as a dictionary
    
    if config_dict['DEVICE'] == 'auto':
        config_dict['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if as_namespace:
        return dict_to_namespace(config_dict)

    return config_dict

def dict_to_namespace(d):
    """
    Recursively converts a dictionary into a SimpleNamespace
    Since our config.yaml file is loaded as a dictionary
    
    It returns:
    SimpleNamespace object
    
    SimpleNamespace provides very lightweight object where we can dynamically add attributes. Eg: ob = SimpleNamespace(); ob.name = 'bob'; ob.age = 23;
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        """
        - loops over each key, value pair in the dictionary
        - recursively calls dict_to_namespace on v, in case its a nested dictionary or list
        - builds a new dictionary with all of the converted values
        - unpacks the dictionary into keyword arguments (**) to build a SimpleNamespace.
        """
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def ensure_path_exists(pth):
    """
    Ensures that a directory at path pth exists
    Creates it (with parent dirs) if it does not exist (exist_ok = True will create parent dictionary)
    """
    if not os.path.exists(pth):
        os.makedirs(pth, exist_ok = True)