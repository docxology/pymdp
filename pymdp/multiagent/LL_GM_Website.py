import os
import sys
from pathlib import Path
import logging
import numpy as np
import copy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    logger.info(f"Added project root to Python path: {project_root}")
    
    # Also add the pymdp package directory specifically
    pymdp_dir = project_root / "pymdp"
    if pymdp_dir.exists():
        sys.path.insert(0, str(pymdp_dir))
        logger.info(f"Added pymdp directory to Python path: {pymdp_dir}")

    # Import core pymdp modules directly from files
    try:
        # Import directly from core files
        from pymdp.utils import obj_array_zeros
        from pymdp.maths import softmax
        logger.info("Successfully imported pymdp modules")
    except ImportError as e:
        # Try importing from local files
        try:
            from utils import obj_array_zeros
            from maths import softmax
            logger.info("Successfully imported local pymdp modules")
        except ImportError as e2:
            logger.error(f"Error importing pymdp modules: {e2}")
            logger.error("Make sure you're in the correct directory and pymdp is properly installed.")
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Python path: {sys.path}")
            sys.exit(1)

    # Import local modules
    try:
        from LL_Methods import get_model_dimensions_from_labels
        logger.info("Successfully imported local modules")
    except ImportError as e:
        logger.error(f"Error importing local modules: {e}")
        logger.error("Make sure all required files are in the multiagent directory.")
        sys.exit(1)

except Exception as e:
    logger.error(f"Error during setup: {e}")
    sys.exit(1)

# Website Agent Labels
_labWeb = {
    'a': {  # actions
        'aᵂᵉᵇ₁': ['NULL_ACT'],
        'aᵂᵉᵇ₂': ['MOBILE_ACT', 'DESKTOP_ACT', 'TABLET_ACT']
    },
    's': {  # states
        'sᶜˢʳ₁': ['BUSINESS_HOURS', 'AFTER_HOURS'],
        'sᶜˢʳ₂': ['EAST', 'CENTRAL', 'WEST']
    },
    's̆': {  # true states
        's̆ᶜˢʳ₁': ['BUSINESS_HOURS', 'AFTER_HOURS'],
        's̆ᶜˢʳ₂': ['EAST', 'CENTRAL', 'WEST']
    },
    'y': {  # observations
        'yᶜˢʳ₁': ['BUSINESS_HOURS_OBS', 'OVERLAP_HOURS_OBS', 'AFTER_HOURS_OBS'],
        'yᶜˢʳ₂': ['EAST_OBS', 'CENTRAL_OBS', 'WEST_OBS'],
        'yᶜˢʳ₃': ['ENGLISH_OBS', 'SPANISH_OBS', 'CHINESE_OBS']
    }
}

# Get model dimensions
_yCsr_car, _yCsr_num, _sCsr_car, _sCsr_num, _aWeb_car, _aWeb_num = get_model_dimensions_from_labels(_labWeb)

# State-Observation (A) Matrix
_Aᵂᵉᵇ = obj_array_zeros([[y_car] + _sCsr_car for y_car in _yCsr_car])

# First observation modality (Ad Schedule)
_Aᵂᵉᵇ[0][
    _labWeb['y']['yᶜˢʳ₁'].index('AFTER_HOURS_OBS'), 
    :, 
    _labWeb['s']['sᶜˢʳ₂'].index('EAST')
] = 1.0
_Aᵂᵉᵇ[0][
    _labWeb['y']['yᶜˢʳ₁'].index('AFTER_HOURS_OBS'), 
    :, 
    _labWeb['s']['sᶜˢʳ₂'].index('CENTRAL')
] = 1.0
_Aᵂᵉᵇ[0][
    _labWeb['y']['yᶜˢʳ₁'].index('BUSINESS_HOURS_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('WEST')
] = 0.8
_Aᵂᵉᵇ[0][
    _labWeb['y']['yᶜˢʳ₁'].index('OVERLAP_HOURS_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('WEST')
] = 0.2

_Aᵂᵉᵇ[0][
    _labWeb['y']['yᶜˢʳ₁'].index('OVERLAP_HOURS_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('WEST')
] = 0.8
_Aᵂᵉᵇ[0][
    _labWeb['y']['yᶜˢʳ₁'].index('BUSINESS_HOURS_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('WEST')
] = 0.2

# Second observation modality (Location Target)
_Aᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('WEST_OBS'), 
    :, 
    _labWeb['s']['sᶜˢʳ₂'].index('EAST')
] = 1.0

_Aᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('WEST_OBS'), 
    :, 
    _labWeb['s']['sᶜˢʳ₂'].index('WEST')
] = 1.0

_BUSINESS_HOURS_MAPPING_WEB = softmax(np.array([1.0, 0]))
_AFTER_HOURS_MAPPING_WEB = softmax(np.array([0.0, 1.0]))

_Aᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('EAST_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('CENTRAL')
] = _BUSINESS_HOURS_MAPPING_WEB[0]
_Aᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('CENTRAL_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('CENTRAL')
] = _BUSINESS_HOURS_MAPPING_WEB[1]

_Aᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('EAST_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('CENTRAL')
] = _AFTER_HOURS_MAPPING_WEB[0]
_Aᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('CENTRAL_OBS'),
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'), 
    _labWeb['s']['sᶜˢʳ₂'].index('CENTRAL')
] = _AFTER_HOURS_MAPPING_WEB[1]

# Third observation modality (Language)
_Aᵂᵉᵇ[2][
    _labWeb['y']['yᶜˢʳ₃'].index('ENGLISH_OBS'), 
    :, 
    _labWeb['s']['sᶜˢʳ₂'].index('EAST')
] = 1.0
_Aᵂᵉᵇ[2][
    _labWeb['y']['yᶜˢʳ₃'].index('SPANISH_OBS'), 
    :, 
    _labWeb['s']['sᶜˢʳ₂'].index('CENTRAL')
] = 1.0
_Aᵂᵉᵇ[2][
    _labWeb['y']['yᶜˢʳ₃'].index('CHINESE_OBS'), 
    :, 
    _labWeb['s']['sᶜˢʳ₂'].index('WEST')
] = 1.0

# State-State Transition (B) Matrix
_Bᵂᵉᵇ = obj_array(_sCsr_num)

# First state factor (Ad Schedule)
_Bᵂᵉᵇ[0] = np.zeros((_sCsr_car[0], _sCsr_car[0], _aWeb_car[0]))

_p_stochWeb = 0.0
_Bᵂᵉᵇ[0][
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'),
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'), 
    _labWeb['a']['aᵂᵉᵇ₁'].index('NULL_ACT')
] = 1.0 - _p_stochWeb
_Bᵂᵉᵇ[0][
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'),
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'), 
    _labWeb['a']['aᵂᵉᵇ₁'].index('NULL_ACT')
] = _p_stochWeb

_Bᵂᵉᵇ[0][
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'),
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'), 
    _labWeb['a']['aᵂᵉᵇ₁'].index('NULL_ACT')
] = 1.0 - _p_stochWeb
_Bᵂᵉᵇ[0][
    _labWeb['s']['sᶜˢʳ₁'].index('BUSINESS_HOURS'),
    _labWeb['s']['sᶜˢʳ₁'].index('AFTER_HOURS'), 
    _labWeb['a']['aᵂᵉᵇ₁'].index('NULL_ACT')
] = _p_stochWeb

# Second state factor (Location Target)
_Bᵂᵉᵇ[1] = np.zeros((_sCsr_car[1], _sCsr_car[1], _aWeb_car[1]))
_Bᵂᵉᵇ[1][
    _labWeb['s']['sᶜˢʳ₂'].index('EAST'), 
    :, 
    _labWeb['a']['aᵂᵉᵇ₂'].index('MOBILE_ACT')
] = 1.0
_Bᵂᵉᵇ[1][
    _labWeb['s']['sᶜˢʳ₂'].index('CENTRAL'), 
    :, 
    _labWeb['a']['aᵂᵉᵇ₂'].index('DESKTOP_ACT')
] = 1.0
_Bᵂᵉᵇ[1][
    _labWeb['s']['sᶜˢʳ₂'].index('WEST'), 
    :, 
    _labWeb['a']['aᵂᵉᵇ₂'].index('TABLET_ACT')
] = 1.0

# Prior preferences (C) Matrix
_Cᵂᵉᵇ = obj_array_zeros([y_car for y_car in _yCsr_car])

_Cᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('EAST_OBS'),
] = 1.0
_Cᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('CENTRAL_OBS'),
] = -1.0
_Cᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('WEST_OBS'),
] = 0.0

# True state matrices
_Ăᵂᵉᵇ = copy.deepcopy(_Aᵂᵉᵇ)
_B̆ᵂᵉᵇ = copy.deepcopy(_Bᵂᵉᵇ)
