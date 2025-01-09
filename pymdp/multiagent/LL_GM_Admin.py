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
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    logger.info(f"Added project root to Python path: {project_root}")
    
    # Import core modules directly
    from ..utils import obj_array_zeros
    from ..maths import softmax
    from .LL_Methods import get_model_dimensions_from_labels
    logger.info("Successfully imported all required modules")

except Exception as e:
    logger.error(f"Error during setup: {e}")
    sys.exit(1)

# Administrator Agent Labels
_labAdm = {
    'a': {  # actions
        'aᴬᵈᵐ₁': ['NULL_ACT'],
        'aᴬᵈᵐ₂': ['BROAD_MATCH_ACT', 'PHRASE_MATCH_ACT', 'EXACT_MATCH_ACT']
    },
    's': {  # states
        'sᵂᵉᵇ₁': ['EXPANDED_TEXT_ADS', 'RESPONSIVE_SEARCH_ADS'],
        'sᵂᵉᵇ₂': ['DESCRIPTION_LINES', 'DISPLAY_URL', 'CALL_TO_ACTION']
    },
    's̆': {  # true states
        's̆ᵂᵉᵇ₁': ['EXPANDED_TEXT_ADS', 'RESPONSIVE_SEARCH_ADS'],
        's̆ᵂᵉᵇ₂': ['DESCRIPTION_LINES', 'DISPLAY_URL', 'CALL_TO_ACTION']
    },
    'y': {  # observations
        'yᵂᵉᵇ₁': ['EXPANDED_TEXT_ADS_OBS', 'RESPONSIVE_SEARCH_ADS_OBS', 'UNKNOWN_OBS'],
        'yᵂᵉᵇ₂': ['DESCRIPTION_LINES_OBS', 'DISPLAY_URL_OBS', 'CALL_TO_ACTION_OBS'],
        'yᵂᵉᵇ₃': ['SITE_LINKS_OBS', 'CALLOUTS_OBS', 'STRUCTURED_SNIPPETS_OBS']
    }
}

# Get model dimensions
_yWeb_car, _yWeb_num, _sWeb_car, _sWeb_num, _aAdm_car, _aAdm_num = get_model_dimensions_from_labels(_labAdm)

# State-Observation (A) Matrix
_Aᴬᵈᵐ = obj_array_zeros([[y_car] + _sWeb_car for y_car in _yWeb_car])

# First observation modality (Ad Type)
_Aᴬᵈᵐ[0][
    _labAdm['y']['yᵂᵉᵇ₁'].index('UNKNOWN_OBS'), 
    :, 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DESCRIPTION_LINES')
] = 1.0
_Aᴬᵈᵐ[0][
    _labAdm['y']['yᵂᵉᵇ₁'].index('UNKNOWN_OBS'), 
    :, 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DISPLAY_URL')
] = 1.0
_Aᴬᵈᵐ[0][
    _labAdm['y']['yᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('CALL_TO_ACTION')
] = 0.8
_Aᴬᵈᵐ[0][
    _labAdm['y']['yᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('CALL_TO_ACTION')
] = 0.2

_Aᴬᵈᵐ[0][
    _labAdm['y']['yᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('CALL_TO_ACTION')
] = 0.8
_Aᴬᵈᵐ[0][
    _labAdm['y']['yᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('CALL_TO_ACTION')
] = 0.2

# Second observation modality (Ad Copy)
_Aᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('CALL_TO_ACTION_OBS'), 
    :, 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DESCRIPTION_LINES')
] = 1.0

_Aᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('CALL_TO_ACTION_OBS'), 
    :, 
    _labAdm['s']['sᵂᵉᵇ₂'].index('CALL_TO_ACTION')
] = 1.0

_EXPANDED_TEXT_ADS_MAPPING_ADM = softmax(np.array([1.0, 0]))
_RESPONSIVE_SEARCH_ADS_MAPPING_ADM = softmax(np.array([0.0, 1.0]))

_Aᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DESCRIPTION_LINES_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DISPLAY_URL')
] = _EXPANDED_TEXT_ADS_MAPPING_ADM[0]
_Aᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DISPLAY_URL_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DISPLAY_URL')
] = _EXPANDED_TEXT_ADS_MAPPING_ADM[1]

_Aᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DESCRIPTION_LINES_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DISPLAY_URL')
] = _RESPONSIVE_SEARCH_ADS_MAPPING_ADM[0]
_Aᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DISPLAY_URL_OBS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'), 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DISPLAY_URL')
] = _RESPONSIVE_SEARCH_ADS_MAPPING_ADM[1]

# Third observation modality (Extensions)
_Aᴬᵈᵐ[2][
    _labAdm['y']['yᵂᵉᵇ₃'].index('SITE_LINKS_OBS'), 
    :, 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DESCRIPTION_LINES')
] = 1.0
_Aᴬᵈᵐ[2][
    _labAdm['y']['yᵂᵉᵇ₃'].index('CALLOUTS_OBS'), 
    :, 
    _labAdm['s']['sᵂᵉᵇ₂'].index('DISPLAY_URL')
] = 1.0
_Aᴬᵈᵐ[2][
    _labAdm['y']['yᵂᵉᵇ₃'].index('STRUCTURED_SNIPPETS_OBS'), 
    :, 
    _labAdm['s']['sᵂᵉᵇ₂'].index('CALL_TO_ACTION')
] = 1.0

# State-State Transition (B) Matrix
_Bᴬᵈᵐ = obj_array(_sWeb_num)

# First state factor (Ad Type)
_Bᴬᵈᵐ[0] = np.zeros((_sWeb_car[0], _sWeb_car[0], _aAdm_car[0]))

_p_stochAdm = 0.0
_Bᴬᵈᵐ[0][
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'), 
    _labAdm['a']['aᴬᵈᵐ₁'].index('NULL_ACT')
] = 1.0 - _p_stochAdm
_Bᴬᵈᵐ[0][
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'), 
    _labAdm['a']['aᴬᵈᵐ₁'].index('NULL_ACT')
] = _p_stochAdm

_Bᴬᵈᵐ[0][
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'), 
    _labAdm['a']['aᴬᵈᵐ₁'].index('NULL_ACT')
] = 1.0 - _p_stochAdm
_Bᴬᵈᵐ[0][
    _labAdm['s']['sᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'),
    _labAdm['s']['sᵂᵉᵇ₁'].index('RESPONSIVE_SEARCH_ADS'), 
    _labAdm['a']['aᴬᵈᵐ₁'].index('NULL_ACT')
] = _p_stochAdm

# Second state factor (Ad Copy Creation)
_Bᴬᵈᵐ[1] = np.zeros((_sWeb_car[1], _sWeb_car[1], _aAdm_car[1]))
_Bᴬᵈᵐ[1][
    _labAdm['s']['sᵂᵉᵇ₂'].index('DESCRIPTION_LINES'), 
    :, 
    _labAdm['a']['aᴬᵈᵐ₂'].index('BROAD_MATCH_ACT')
] = 1.0
_Bᴬᵈᵐ[1][
    _labAdm['s']['sᵂᵉᵇ₂'].index('DISPLAY_URL'), 
    :, 
    _labAdm['a']['aᴬᵈᵐ₂'].index('PHRASE_MATCH_ACT')
] = 1.0
_Bᴬᵈᵐ[1][
    _labAdm['s']['sᵂᵉᵇ₂'].index('CALL_TO_ACTION'), 
    :, 
    _labAdm['a']['aᴬᵈᵐ₂'].index('EXACT_MATCH_ACT')
] = 1.0

# Prior preferences (C) Matrix
_Cᴬᵈᵐ = obj_array_zeros([y_car for y_car in _yWeb_car])

_Cᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DESCRIPTION_LINES_OBS'),
] = 1.0
_Cᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DISPLAY_URL_OBS'),
] = -1.0
_Cᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('CALL_TO_ACTION_OBS'),
] = 0.0

# Control factor indices
_control_fac_idx_Adm = [1]

# True state matrices
_Ăᴬᵈᵐ = copy.deepcopy(_Aᴬᵈᵐ)
_B̆ᴬᵈᵐ = copy.deepcopy(_Bᴬᵈᵐ)