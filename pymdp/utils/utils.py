import numpy as np
import logging
from typing import List, Union, Any

logger = logging.getLogger(__name__)

def obj_array(shape: Union[int, tuple]) -> np.ndarray:
    """
    Creates a numpy object array with specified shape
    
    Parameters
    ----------
    shape : int or tuple
        Shape of the desired array
    
    Returns
    -------
    np.ndarray
        Object array of given shape
    """
    logger.debug(f"Creating object array with shape: {shape}")
    return np.empty(shape, dtype=object)

def obj_array_zeros(shape_list: List[List[int]]) -> np.ndarray:
    """
    Creates a numpy object array with subarrays of zeros
    
    Parameters
    ----------
    shape_list : list of lists
        List of shapes for the subarrays
    
    Returns
    -------
    np.ndarray
        Object array containing zero arrays of specified shapes
    """
    logger.debug(f"Creating zero-initialized object array with shapes: {shape_list}")
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.zeros(shape)
    return arr

def sample(arr: Union[np.ndarray, List[float]]) -> int:
    """
    Sample from a probability distribution
    
    Parameters
    ----------
    arr : np.ndarray or list
        Probability distribution to sample from
    
    Returns
    -------
    int
        Sampled index
    """
    # Convert list to numpy array if needed
    if isinstance(arr, list):
        arr = np.array(arr)
    
    # Ensure array is properly normalized
    arr = arr / np.sum(arr)
    
    logger.debug(f"Sampling from distribution with shape {arr.shape}")
    return np.random.choice(arr.size, p=arr.flatten())

def to_obj_array(arr: Union[List[Any], np.ndarray]) -> np.ndarray:
    """
    Convert a list or array into an object array
    
    Parameters
    ----------
    arr : list or np.ndarray
        List/array to convert to object array
    
    Returns
    -------
    np.ndarray
        Object array containing the input elements
    """
    logger.debug(f"Converting to object array, input type: {type(arr)}")
    
    try:
        # Handle numpy array input
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                logger.debug("Input is already an object array")
                return arr
            else:
                logger.debug("Converting numpy array to object array")
                obj_arr = obj_array(len(arr))
                for i in range(len(arr)):
                    obj_arr[i] = arr[i]
                return obj_arr
        
        # Handle list input
        elif isinstance(arr, list):
            logger.debug(f"Converting list of length {len(arr)} to object array")
            obj_arr = obj_array(len(arr))
            for i, item in enumerate(arr):
                # Convert nested lists/arrays
                if isinstance(item, (list, np.ndarray)):
                    obj_arr[i] = np.array(item)
                    logger.debug(f"Converted item {i} from {type(item)} to numpy array")
                else:
                    # Wrap scalar values in 0-d arrays
                    obj_arr[i] = np.array([item])
                    logger.debug(f"Wrapped scalar item {i} in numpy array")
            return obj_arr
            
        else:
            raise TypeError(f"Cannot convert type {type(arr)} to object array")
            
    except Exception as e:
        logger.error(f"Error converting to object array: {str(e)}")
        logger.error(f"Input was: {arr}")
        raise ValueError(f"Could not convert input to object array: {str(e)}")

def obj_array_uniform(shapes: List[Union[int, tuple]]) -> np.ndarray:
    """
    Create object array with uniform distributions
    
    Parameters
    ----------
    shapes : list
        List of shapes for each array
    
    Returns
    -------
    np.ndarray
        Object array containing uniform distributions
    """
    logger.debug(f"Creating uniform object array with shapes: {shapes}")
    
    try:
        obj_arr = obj_array(len(shapes))
        for i, shape in enumerate(shapes):
            obj_arr[i] = np.ones(shape) / np.prod(shape)
            logger.debug(f"Created uniform distribution with shape {shape}")
        return obj_arr
        
    except Exception as e:
        logger.error(f"Error creating uniform object array: {str(e)}")
        raise

def is_obj_array(arr: Any) -> bool:
    """
    Check if input is a numpy object array
    
    Parameters
    ----------
    arr : any
        Input to check
    
    Returns
    -------
    bool
        True if input is a numpy object array
    """
    # First check if it's a numpy array
    if not isinstance(arr, np.ndarray):
        return False
    return arr.dtype == object

def ensure_array(arr: Union[List, np.ndarray]) -> np.ndarray:
    """
    Ensure input is a numpy array
    
    Parameters
    ----------
    arr : list or np.ndarray
        Input to convert
    
    Returns
    -------
    np.ndarray
        Input converted to numpy array
    """
    if not isinstance(arr, np.ndarray):
        return np.array(arr)
    return arr 