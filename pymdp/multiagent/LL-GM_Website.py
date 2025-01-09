_labWeb = { ## labels for Website interaction
    "a": {
        "aᵂᵉᵇ₁": [ ## "NULL"
            "NULL_ACT",
        ],
        "aᵂᵉᵇ₂": [ ## "SET_DEVICE_TARGET_ACTION"
            "MOBILE_ACT",
            "DESKTOP_ACT",
            "TABLET_ACT"
        ],
    },
    "s": {
        "sᶜˢʳ₁": [ ## "AD_SCHEDULE"
            "BUSINESS_HOURS", 
            "AFTER_HOURS",
        ],
        "sᶜˢʳ₂": [ ## "LOCATION_TARGET"
            "EAST", 
            "CENTRAL",
            "WEST"
        ],
    },
    "s̆": {
        "s̆ᶜˢʳ₁": [ ## "AD_SCHEDULE"
            "BUSINESS_HOURS", 
            "AFTER_HOURS",
        ],
        "s̆ᶜˢʳ₂": [ ## "LOCATION_TARGET"
            "EAST", 
            "CENTRAL",
            "WEST"
        ],
    },
    "y": {
        "yᶜˢʳ₁": [ ## "AD_SCHEDULE_OBS"
            "BUSINESS_HOURS_OBS",
            "OVERLAP_HOURS_OBS",
            "AFTER_HOURS_OBS"
        ],
        "yᶜˢʳ₂": [ ## "LOCATION_TARGET_OBS"
            "EAST_OBS",
            "CENTRAL_OBS",
            "WEST_OBS"
        ],
        "yᶜˢʳ₃": [ ## "LANGUAGE_OBS"
            "ENGLISH_OBS",
            "SPANISH_OBS",
            "CHINESE_OBS"
        ],
    },
}
_yCsr_car,_yCsr_num, _sCsr_car,_sCsr_num, _aWeb_car,_aWeb_num = get_model_dimensions_from_labels(_labWeb)
_yCsr_car,_yCsr_num, _sCsr_car,_sCsr_num, _aWeb_car,_aWeb_num


print(f'{_aWeb_car=}') ## cardinality of control factors
print(f'{_aWeb_num=}') ## number of control factors

print(f'{_sCsr_car=}') ## cardinality of state factors
print(f'{_sCsr_num=}') ## number of state factors

print(f'{_yCsr_car=}') ## cardinality of observation modalities
print(f'{_yCsr_num=}') ## number of observation modalities

_aWeb_fac_names = list(_labWeb['a'].keys()); print(f'{_aWeb_fac_names=}') ## control factor names
_sCsr_fac_names = list(_labWeb['s'].keys()); print(f'{_sCsr_fac_names=}') ## state factor names
_s̆Csr_fac_names = list(_labWeb['s̆'].keys()); print(f'{_s̆Csr_fac_names=}') ## state factor names
_yCsr_mod_names = list(_labWeb['y'].keys()); print(f'{_yCsr_mod_names=}') ## observation modality names

_Aᵂᵉᵇ = utils.obj_array_zeros([[y_car] + _sCsr_car for y_car in _yCsr_car])
print(f'{len(_Aᵂᵉᵇ)=}')
_Aᵂᵉᵇ

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

_Aᵂᵉᵇ[0]

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

_Aᵂᵉᵇ[1]

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

_Aᵂᵉᵇ[2]

print(f'=== _sCsr_car:\n{_sCsr_car}')
print(f'=== _yCsr_car:\n{_yCsr_car}')
_Aᵂᵉᵇ

_Bᵂᵉᵇ = utils.obj_array(_sCsr_num); print(f'{_sCsr_num=}')
print(f'{len(_Bᵂᵉᵇ)=}')
_Bᵂᵉᵇ

_Bᵂᵉᵇ[0] = np.zeros((_sCsr_car[0], _sCsr_car[0], _aWeb_car[0])); print(f'{_sCsr_car[0]=}, {_sCsr_car[0]=}, {_aWeb_car[0]=}')
_Bᵂᵉᵇ[0]

_p_stochWeb = 0.0

## we cannot influence factor zero, set up the 'default' stationary dynamics - 
## one state just maps to itself at the next timestep with very high probability, 
## by default. So this means the AD_SCHEDULE state can change from one to another with 
## some low probability (p_stoch)

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

_Bᵂᵉᵇ[0]

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

_Bᵂᵉᵇ[1]

print(f'=== _aWeb_car:\n{_aWeb_car}')
print(f'=== _sCsr_car:\n{_sCsr_car}')
_Bᵂᵉᵇ

_Cᵂᵉᵇ = utils.obj_array_zeros([y_car for y_car in _yCsr_car])
_Cᵂᵉᵇ

_Cᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('EAST_OBS'),
] = 1.0
_Cᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('CENTRAL_OBS'),
] = -1.0
_Cᵂᵉᵇ[1][
    _labWeb['y']['yᶜˢʳ₂'].index('WEST_OBS'),
] = 0.0

_Cᵂᵉᵇ[1]

_agtWeb = Agent(
    A=_Aᵂᵉᵇ, 
    B=_Bᵂᵉᵇ, 
    C=_Cᵂᵉᵇ, 
)
_agtWeb

# This is the Website agent’s generative process for its environment.
# It is important to note that the generative process doesn’t have to be described by A and B matrices - it can just be the arbitrary ‘rules of the game’ that you ‘write in’ as a modeller. But here we just use the same transition/likelihood matrices to make the sampling process straightforward.

## observation/transition matrices characterising the generative process
_Ăᶜˢʳ = copy.deepcopy(_Aᵂᵉᵇ)
_B̆ᶜˢʳ = copy.deepcopy(_Bᵂᵉᵇ)

