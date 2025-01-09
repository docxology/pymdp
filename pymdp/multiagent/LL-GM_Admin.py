 
_labAdm = { ## labels for Administrator
    "a": {
        "aᴬᵈᵐ₁": [ ## "NULL"
            "NULL_ACT", 
        ],
        "aᴬᵈᵐ₂": [ ## "SET_MATCH_TYPE_ACTION"
            "BROAD_MATCH_ACT", 
            "PHRASE_MATCH_ACT", 
            "EXACT_MATCH_ACT"
        ],
    },
    "s": {
        "sᵂᵉᵇ₁": [ ## "AD_TYPE"
            "EXPANDED_TEXT_ADS", 
            "RESPONSIVE_SEARCH_ADS",
        ],
        "sᵂᵉᵇ₂": [ ## "AD_COPY_CREATION"
            "DESCRIPTION_LINES", 
            "DISPLAY_URL", 
            "CALL_TO_ACTION"
        ],
        ## LATER:
        # "sᵂᵉᵇ₃": [ ## "AD_EXTENSIONS"
        #     "SITE_LINKS", 
        #     "CALLOUTS",             
        #     "STRUCTURED_SNIPPETS"
        # ],
        # "sᶜˢʳ₁": [ ## "FEEDBACK"
        #     "POSITIVE", 
        #     "NEGATIVE"
        # ],
    },
    "s̆": {
        "s̆ᵂᵉᵇ₁": [ ## "AD_TYPE"
            "EXPANDED_TEXT_ADS", 
            "RESPONSIVE_SEARCH_ADS",
        ],
        "s̆ᵂᵉᵇ₂": [ ## "AD_COPY_CREATION"
            "DESCRIPTION_LINES", 
            "DISPLAY_URL", 
            "CALL_TO_ACTION"
        ],
    },    
    "y": {
        "yᵂᵉᵇ₁": [ ## "AD_TYPE_OBS"
            "EXPANDED_TEXT_ADS_OBS",
            "RESPONSIVE_SEARCH_ADS_OBS",
            "UNKNOWN_OBS"
        ],
        "yᵂᵉᵇ₂": [ ## "AD_COPY_CREATION_OBS"
            "DESCRIPTION_LINES_OBS",
            "DISPLAY_URL_OBS",
            "CALL_TO_ACTION_OBS"
        ],
        "yᵂᵉᵇ₃": [ ## "AD_EXTENSIONS_OBS"
            "SITE_LINKS_OBS",
            "CALLOUTS_OBS",
            "STRUCTURED_SNIPPETS_OBS"
        ],
        ## LATER:
        # "yᶜˢʳ₁": [ ## "FEEDBACK_OBS"
        #     "POSITIVE_OBS",
        #     "NEGATIVE_OBS"
        # ]
    },
}
_yWeb_car,_yWeb_num, _sWeb_car,_sWeb_num, _aAdm_car,_aAdm_num = get_model_dimensions_from_labels(_labAdm) ##.
_yWeb_car,_yWeb_num, _sWeb_car,_sWeb_num, _aAdm_car,_aAdm_num




print(f'{_aAdm_car=}') ## cardinality of control factors
print(f'{_aAdm_num=}') ## number of control factors

print(f'{_sWeb_car=}') ## cardinality of state factors
print(f'{_sWeb_num=}') ## number of state factors

print(f'{_yWeb_car=}') ## cardinality of observation modalities
print(f'{_yWeb_num=}') ## number of observation modalities

_aAdm_fac_names = list(_labAdm['a'].keys()); print(f'{_aAdm_fac_names=}') ## control factor names
_sWeb_fac_names = list(_labAdm['s'].keys()); print(f'{_sWeb_fac_names=}') ## state factor names
_s̆Web_fac_names = list(_labAdm['s̆'].keys()); print(f'{_s̆Web_fac_names=}') ## state factor names
_yWeb_mod_names = list(_labAdm['y'].keys()); print(f'{_yWeb_mod_names=}') ## observation modality names

## A = utils.obj_array_zeros([[o] + _car_state for _, o in enumerate(_car_obser)]) ##.
_Aᴬᵈᵐ = utils.obj_array_zeros([[y_car] + _sWeb_car for y_car in _yWeb_car])
print(f'{len(_Aᴬᵈᵐ)=}')
_Aᴬᵈᵐ

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

_Aᴬᵈᵐ[0]

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

_Aᴬᵈᵐ[1]

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

_Aᴬᵈᵐ[2]

print(f'=== _sWeb_car:\n{_sWeb_car}')
print(f'=== _yWeb_car:\n{_yWeb_car}')
_Aᴬᵈᵐ

_control_fac_idx_Adm = [1] ## used in Agent constructor

_Bᴬᵈᵐ = utils.obj_array(_sWeb_num); print(f'{_sWeb_num=}')
print(f'{len(_Bᴬᵈᵐ)=}')

_Bᴬᵈᵐ

_Bᴬᵈᵐ[0] = np.zeros((_sWeb_car[0], _sWeb_car[0], _aAdm_car[0])); print(f'{_sWeb_car[0]=}, {_sWeb_car[0]=}, {_aAdm_car[0]=}')
_Bᴬᵈᵐ[0]

_p_stochAdm = 0.0
## we cannot influence factor zero, set up the 'default' stationary dynamics - 
## one state just maps to itself at the next timestep with very high probability, 
## by default. So this means the AD_TYPE state can change from one to another with 
## some low probability (p_stoch)
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

_Bᴬᵈᵐ[0]

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

_Bᴬᵈᵐ[1]

print(f'=== _aAdm_car:\n{_aAdm_car}')
print(f'=== _sWeb_car:\n{_sWeb_car}')
_Bᴬᵈᵐ

_Cᴬᵈᵐ = utils.obj_array_zeros([y_car for y_car in _yWeb_car])
_Cᴬᵈᵐ

_Cᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DESCRIPTION_LINES_OBS'),
] = 1.0
_Cᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('DISPLAY_URL_OBS'),
] = -1.0
_Cᴬᵈᵐ[1][
    _labAdm['y']['yᵂᵉᵇ₂'].index('CALL_TO_ACTION_OBS'),
] = 0.0

_Cᴬᵈᵐ[1]

_agtAdm = Agent(
    A=_Aᴬᵈᵐ, 
    B=_Bᴬᵈᵐ, 
    C=_Cᴬᵈᵐ, 
    control_fac_idx=_control_fac_idx_Adm
)
_agtAdm

## observation/transition matrices characterising the generative process

## currently only true values of Website
##   should be _Ăᴬᵈᵐ to include true values of complete Administrator env
_Ăᵂᵉᵇ = copy.deepcopy(_Aᴬᵈᵐ)

## True next-state may be calculated without a B matrix
## currently only true values of Website
##   should be _B̆ᴬᵈᵐ to include true values of complete Administrator env
_B̆ᵂᵉᵇ = copy.deepcopy(_Bᴬᵈᵐ)