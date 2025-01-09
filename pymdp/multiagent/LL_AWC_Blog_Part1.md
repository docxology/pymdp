Author
Kobus Esterhuysen

Published
November 21, 2024

Modified
January 2, 2025

TOC
1 Agent interactions
2 Player/Game interaction
2.1 Player agent
2.1.1 State factors
2.1.1.1 
 (GAME_STATE)
2.1.1.2 
 (PLAYING_VS_SAMPLING)
2.1.2 Observation modalities
2.1.2.1 
 (Observations of the game state, GAME_STATE_OBS)
2.1.2.2 
 (Reward observations, GAME_OUTCOME)
2.1.2.3 
 (“Proprioceptive” or self-state observations, ACTION_SELF_OBS)
Note about the arbitrariness of ‘labelling’ observations, before defining the A and C matrices.
2.1.3 Control factors
2.1.3.1 
 (NULL)
2.1.3.2 
 (PLAYING_VS_SAMPLING_CONTROL)
(Controllable-) Transition Dynamics
Prior preferences
Initialise an instance of the Agent() class:
2.2 Game environment
3 Game/Config interaction
3.1 Game agent
3.1.1 State factors
3.1.1.1 
 (GAME_STATE_INVERSION)
3.1.2 Observation modalities
3.1.2.1 
 (Observations of the game state, GAME_STATE_INVERSION_OBS)
3.1.3 Control factors
3.1.3.1 
 (REWARD_INVERSION_CONTROL)
Prior preferences
Initialise an instance of the Agent() class:
3.2 Config environment
4 Run simulation

Back to Portfolio of Projects |  LearnableLoopAI.com |  Blog |  LinkedIn


In Part 1 we setup a three-entity multi-agent system that use PyMDP. However, we make use of the sequence structure of RxInfer. To do this we modify an existing demo:

https://github.com/infer-actively/pymdp/blob/master/examples/agent_demo.ipynb

The modifications are:

Do some restructuring of the contents

Use some of my preferred symbols

Use the _car prefix (for cardinality of factors and modalities)

Add a _lab dict for all labels

Use implied label indices from _lab to set values of matrices

Add visualization

Prefix globals with _

Lines where changes were made usually contains ##.

structure highest level of content based on interaction pairs

use math symbols with superscripts indicating agent identity

change sta –> s in _lab dict

change obs –> o in _lab dict

change ctr –> u in _lab dict

reorder main loop steps to act, future, next, observe, infer, slide to align with the approach used by RxInfer

change idx –> val for value

partition _lab dict to separate agt & sus labels

add a third entity called Config

add a game/config interaction

In Part 1 we will not be concerned with good training outcomes and performance. The emphasis will be on setting up a working structure for this multi-agent problem that will serve as a foundation for a client problem related to the entities:

Administrator (role of agent)
Website (role of both environment and agent)
Consumer (role of environment)
Migration of the current structure to support this client problem will be undertaken in Part 2.

import os
import sys
import pathlib
import numpy as np
## import seaborn as sns
import matplotlib.pyplot as plt
import copy
## from pprint import pprint ##.

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import softmax

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

## ?utils.get_model_dimensions_from_labels
##.rewrite method to allow for shorter names so that the _lab dict can be used more
##   easily when setting up the matrices
def get_model_dimensions_from_labels(model_labels):
    ## modalities = model_labels['observations']
    modalities = model_labels['o'] ##.
    num_modalities = len(modalities.keys())
    num_obs = [len(modalities[modality]) for modality in modalities.keys()]

    ## factors = model_labels['states']
    factors = model_labels['s'] ##.
    num_factors = len(factors.keys())
    num_states = [len(factors[factor]) for factor in factors.keys()]

    ## if 'actions' in model_labels.keys():
    if 'u' in model_labels.keys(): ##.
        ## controls = model_labels['actions']
        controls = model_labels['u'] ##.
        num_control_fac = len(controls.keys())
        num_controls = [len(controls[cfac]) for cfac in controls.keys()]
        return num_obs, num_modalities, num_states, num_factors, num_controls, num_control_fac
    else:
        return num_obs, num_modalities, num_states, num_factors

1 Agent interactions
The following diagram is a representation of the interactions between the entities in this project.



The Player entity (agent) has symbols:

 is the inferred action states of the Game entity
 is the inferred parameters of the Game entity
 is the inferred state of the Game entity
 is the predicted observation of the Game entity
The Game entity (envir) has symbols:

 is the action on the Game entity
 is the true parameters of the Game entity
 is the true state of the Game entity
 is the exogenous information impacting the Game entity
 is the observation from the Game entity
The Game entity (agent) has symbols:

 is the inferred action states of the Config entity
 is the inferred parameters of the Config entity
 is the inferred state of the Config entity
 is the predicted observation of the Config entity
The Config entity (envir) has symbols:

 is the action on the Config entity
 is the true parameters of the Config entity
 is the true state of the Config entity
 is the exogenous information impacting the Config entity
 is the observation from the Config entity
2 Player/Game interaction
2.1 Player agent
This is the agent’s generative model for the game environment which embodies the system-under-steer for the player/game interaction.

2.1.1 State factors
We assume the agent’s “represents” (this should make you think: generative model , not process ) its environment using two latent variables that are statistically independent of one another - we can thus represent them using two hidden state factors.

We refer to these two hidden state factors as

 (GAME_STATE)
 (PLAYING_VS_SAMPLING)
2.1.1.1 
 (GAME_STATE)
The first factor is a binary variable representing some ‘reward structure’ that characterises the world. It has two possible values or levels:

one level that will lead to rewards with high probability
, a state/level we will call HIGH_REW, and
another level that will lead to “punishments” (e.g. losing money) with high probability
, a state/level we will call LOW_REW
You can think of this hidden state factor as describing the ‘pay-off’ structure of e.g. a two-armed bandit or slot-machine with two different settings - one where you’re more likely to win (HIGH_REW), and one where you’re more likely to lose (LOW_REW). Crucially, the agent doesn’t know what the GAME_STATE actually is. They will have to infer it by actively furnishing themselves with observations

2.1.1.2 
 (PLAYING_VS_SAMPLING)
The second factor is a ternary (3-valued) variable representing the decision-state or ‘sampling state’ of the agent itself.

The first state/level of this hidden state factor is just the
starting or initial state of the agent
, a state that we can call START
the second state/level is the
state the agent occupies when “playing” the multi-armed bandit or slot machine
, a state that we can call PLAYING
the third state/level of this factor is a
“sampling state”
, a state that we can call SAMPLING
This is a decision-state that the agent occupies when it is “sampling” data in order to find out the level of the first hidden state factor - the GAME_STATE 
.
2.1.2 Observation modalities
The observation modalities themselves are divided into 3 modalities. You can think of these as 3 independent sources of information that the agent has access to. You could think of this in direct perceptual terms - e.g. 3 different sensory organs like eyes, ears, & nose, that give you qualitatively-different kinds of information. Or you can think of it more abstractly - like getting your news from 3 different media sources (online news articles, Twitter feed, and Instagram).

2.1.2.1 
 (Observations of the game state, GAME_STATE_OBS)
The first observation modality is the 
 (GAME_STATE_OBS) modality, and corresponds to observations that give the agent information about the GAME_STATE 
. There are three possible outcomes within this modality:

HIGH_REW_EVIDENCE
LOW_REW_EVIDENCE
NO_EVIDENCE
So the first outcome can be described as lending evidence to the idea that the GAME_STATE 
 is HIGH_REW; the second outcome can be described as lending evidence to the idea that the GAME_STATE 
 is LOW_REW; and the third outcome within this modality doesn’t tell the agent one way or another whether the GAME_STATE 
 is HIGH_REW or LOW_REW.

2.1.2.2 
 (Reward observations, GAME_OUTCOME)
The second observation modality is the 
 (GAME_OUTCOME) modality, and corresponds to arbitrary observations that are functions of the GAME_STATE 
. We call the first outcome level of this modality

REWARD
, which gives you a hint about how we’ll set up the C matrix (the agent’s “utility function” over outcomes). We call the second outcome level of this modality
PUN
 = 1, and the third outcome level
NEUTRAL
By design, we will set up the A matrix such that the REWARD outcome is (expected to be) more likely when the GAME_STATE 
 is HIGH_REW (0) and when the agent is in the PLAYING state, and that the PUN outcome is (expected to be) more likely when the GAME_STATE 
 is LOW_REW (1) and the agent is in the PLAYING state. The NEUTRAL outcome is not expected to occur when the agent is playing the game, but will be expected to occur when the agent is in the SAMPLING state. This NEUTRAL outcome within the 
 (GAME_OUTCOME) modality is thus a meaningless or ‘null’ observation that the agent gets when it’s not actually playing the game (because an observation has to be sampled nonetheless from all modalities).

2.1.2.3 
 (“Proprioceptive” or self-state observations, ACTION_SELF_OBS)
The third observation modality is the 
 (ACTION_SELF_OBS) modality, and corresponds to the agent observing what level of the 
 (PLAYING_VS_SAMPLING) state it is currently in. These observations are direct, ‘unambiguous’ mappings to the true 
 (PLAYING_VS_SAMPLING) state, and simply allow the agent to “know” whether it’s playing the game, sampling information to learn about the game state, or where it’s sitting at the START state. The levels of this outcome are simply thus

START_O,
PLAY_O, and
SAMPLE_O,
where the _O suffix simply distinguishes them from their corresponding hidden states, for which they provide direct evidence.

Note about the arbitrariness of ‘labelling’ observations, before defining the A and C matrices.
There is a bit of a circularity here, in that that we’re “pre-empting” what the A matrix (likelihood mapping) should look like, by giving these observations labels that imply particular roles or meanings. An observation per se doesn’t mean anything, it’s just some discrete index that distinguishes it from another observation. It’s only through its probabilistic relationship to hidden states (encoded in the A matrix, as we’ll see below) that we endow an observation with meaning. For example: by already labelling 
 (GAME_STATE_OBS) as HIGH_REW_EVIDENCE, that’s a hint about how we’re going to structure the A matrix for the 
 (GAME_STATE_OBS) modality.

2.1.3 Control factors
The ‘control state’ factors are the agent’s representation of the control states (or actions) that it believes can influence the dynamics of the hidden states - i.e. hidden state factors that are under the influence of control states are are ‘controllable’. In practice, we often encode every hidden state factor as being under the influence of control states, but the ‘uncontrollable’ hidden state factors are driven by a trivially-1-dimensional control state or action-affordance. This trivial action simply ‘maintains the default environmental dynamics as they are’ i.e. does nothing. This will become more clear when we set up the transition model (the B matrices) below.

2.1.3.1 
 (NULL)
This reflects the agent’s lack of ability to influence the GAME_STATE 
 using policies or actions. The dimensionality of this control factor is 1, and there is only one action along this control factor:

NULL_ACTION or “don’t do anything to do the environment”.
This just means that the transition dynamics along the GAME_STATE 
 hidden state factor have their own, uncontrollable dynamics that are not conditioned on this 
 (NULL) control state - or rather, always conditioned on an unchanging, 1-dimensional NULL_ACTION.

2.1.3.2 
 (PLAYING_VS_SAMPLING_CONTROL)
This is a control factor that reflects the agent’s ability to move itself between the START, PLAYING and SAMPLING states of the 
 (PLAYING_VS_SAMPLING) hidden state factor. The levels/values of this control factor are

START_ACTION
PLAY_ACTION
SAMPLE_ACTION
When we describe the B matrices below, we will set up the transition dynamics of the 
 (PLAYING_VS_SAMPLING) hidden state factor, such that they are totally determined by the value of the 
 (PLAYING_VS_SAMPLING_CONTROL) factor.

(Controllable-) Transition Dynamics
Importantly, some hidden state factors are controllable by the agent, meaning that the probability of being in state 
 at 
 isn’t merely a function of the state at 
, but also of actions (or from the generative model’s perspective, control states ). So each transition likelihood or B matrix encodes conditional probability distributions over states at 
, where the conditioning variables are both the states at 
 and the actions at 
. This extra conditioning on control states is encoded by a third, lagging dimension on each factor-specific B matrix. So they are technically B “tensors” or an array of action-conditioned B matrices.

For example, in our case the 2nd hidden state factor 
 (PLAYING_VS_SAMPLING) is under the control of the agent, which means the corresponding transition likelihoods B[1] are index-able by both previous state and action.

_labPlrGam = { ## labels for Player/Game (agent/environment) interaction
    ## agt
    "u": {
        "uᴳᵃᵐ₁": [ ## "NULL"
            "NULL_ACTION", 
        ],
        "uᴳᵃᵐ₂": [ ## "PLAYING_VS_SAMPLING_CONTROL"
            "START_ACTION", 
            "PLAY_ACTION", 
            "SAMPLE_ACTION"
        ],
    },
    "s": {
        "sᴳᵃᵐ₁": [ ## "GAME_STATE"
            "HIGH_REW", 
            "LOW_REW"
        ],
        "sᴳᵃᵐ₂": [ ## "PLAYING_VS_SAMPLING"
            "START", 
            "PLAYING", 
            "SAMPLING"
        ],
    },
    "o": {
        "oᴳᵃᵐ₁": [ ## "GAME_STATE_OBS"
            "HIGH_REW_EVIDENCE",
            "LOW_REW_EVIDENCE",
            "NO_EVIDENCE"            
        ],
        "oᴳᵃᵐ₂": [ ## "GAME_OUTCOME"
            "REWARD",
            "PUN",
            "NEUTRAL"
        ],
        "oᴳᵃᵐ₃": [ ## "ACTION_SELF_OBS", direct obser of hidden state PLAYING_VS_SAMPLING
            "START_O",
            "PLAY_O",
            "SAMPLE_O"
        ]
    },
    ## env/sus
    "a": { 
        "aᴳᵃᵐ₁": [ ## "NULL"
            "NULL_ACTION", 
        ],
        "aᴳᵃᵐ₂": [ ## "PLAYING_VS_SAMPLING_CONTROL"
            "START_ACTION", 
            "PLAY_ACTION", 
            "SAMPLE_ACTION"
        ],
    },
    "s̆": {
        "s̆ᴳᵃᵐ₁": [ ## "GAME_STATE"
            "HIGH_REW", 
            "LOW_REW"
        ],
        "s̆ᴳᵃᵐ₂": [ ## "PLAYING_VS_SAMPLING"
            "START", 
            "PLAYING", 
            "SAMPLING"
        ],
    },    
    "y": {
        "yᴳᵃᵐ₁": [ ## "GAME_STATE_OBS"
            "HIGH_REW_EVIDENCE",
            "LOW_REW_EVIDENCE",
            "NO_EVIDENCE"            
        ],
        "yᴳᵃᵐ₂": [ ## "GAME_OUTCOME"
            "REWARD",
            "PUN",
            "NEUTRAL"
        ],
        "yᴳᵃᵐ₃": [ ## "ACTION_SELF_OBS", direct obser of hidden state PLAYING_VS_SAMPLING
            "START_O",
            "PLAY_O",
            "SAMPLE_O"
        ]
    },  
}
_oGam_car,_oGam_num, _sGam_car,_sGam_num, _uGam_car,_uGam_num = get_model_dimensions_from_labels(_labPlrGam) ##.
_oGam_car,_oGam_num, _sGam_car,_sGam_num, _uGam_car,_uGam_num

([3, 3, 3], 3, [2, 3], 2, [1, 3], 2)
print(f'{_uGam_car=}') ##.cardinality of control factors
print(f'{_uGam_num=}') ##.number of control factors

print(f'{_sGam_car=}') ##.cardinality of state factors
print(f'{_sGam_num=}') ##.number of state factors

print(f'{_oGam_car=}') ##.cardinality of observation modalities
print(f'{_oGam_num=}') ##.number of observation modalities

_uGam_car=[1, 3]
_uGam_num=2
_sGam_car=[2, 3]
_sGam_num=2
_oGam_car=[3, 3, 3]
_oGam_num=3
##.
_uGam_fac_names = list(_labPlrGam['u'].keys()); print(f'{_uGam_fac_names=}') ##.control factor names
_sGam_fac_names = list(_labPlrGam['s'].keys()); print(f'{_sGam_fac_names=}') ##.state factor names
_oGam_mod_names = list(_labPlrGam['o'].keys()); print(f'{_oGam_mod_names=}') ##.observation modality names

_uGam_fac_names=['uᴳᵃᵐ₁', 'uᴳᵃᵐ₂']
_sGam_fac_names=['sᴳᵃᵐ₁', 'sᴳᵃᵐ₂']
_oGam_mod_names=['oᴳᵃᵐ₁', 'oᴳᵃᵐ₂', 'oᴳᵃᵐ₃']
##.
_aGam_fac_names = list(_labPlrGam['a'].keys()); print(f'{_aGam_fac_names=}') ##.control factor names
_s̆Gam_fac_names = list(_labPlrGam['s̆'].keys()); print(f'{_s̆Gam_fac_names=}') ##.state factor names
_yGam_mod_names = list(_labPlrGam['y'].keys()); print(f'{_yGam_mod_names=}') ##.observation modality names

_aGam_fac_names=['aᴳᵃᵐ₁', 'aᴳᵃᵐ₂']
_s̆Gam_fac_names=['s̆ᴳᵃᵐ₁', 's̆ᴳᵃᵐ₂']
_yGam_mod_names=['yᴳᵃᵐ₁', 'yᴳᵃᵐ₂', 'yᴳᵃᵐ₃']
Setting up observation likelihood matrix - first main component of generative model

## A = utils.obj_array_zeros([[o] + _car_state for _, o in enumerate(_car_obser)]) ##.
_Aᴾˡʳᴳᵃᵐ = utils.obj_array_zeros([[o_car] + _sGam_car for o_car in _oGam_car]) ##.
print(f'{len(_Aᴾˡʳᴳᵃᵐ)=}')
_Aᴾˡʳᴳᵃᵐ

len(_Aᴾˡʳᴳᵃᵐ)=3
array([array([[[0., 0., 0.],
               [0., 0., 0.]],

              [[0., 0., 0.],
               [0., 0., 0.]],

              [[0., 0., 0.],
               [0., 0., 0.]]]), array([[[0., 0., 0.],
                                        [0., 0., 0.]],

                                       [[0., 0., 0.],
                                        [0., 0., 0.]],

                                       [[0., 0., 0.],
                                        [0., 0., 0.]]]),
       array([[[0., 0., 0.],
               [0., 0., 0.]],

              [[0., 0., 0.],
               [0., 0., 0.]],

              [[0., 0., 0.],
               [0., 0., 0.]]])], dtype=object)
Set up the first modality’s likelihood mapping, correspond to how 
 (GAME_STATE_OBS) are related to hidden states.

## they always get the 'no evidence' observation in the START STATE
_Aᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₁'].index('NO_EVIDENCE'), 
    :, 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('START')
] = 1.0
## they always get the 'no evidence' observation in the PLAYING STATE
_Aᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₁'].index('NO_EVIDENCE'), 
    :, 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('PLAYING')
] = 1.0
## the agent expects to see the HIGH_REW_EVIDENCE observation with 80% probability, 
##   if the GAME_STATE is HIGH_REW, and the agent is in the SAMPLING state
_Aᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₁'].index('HIGH_REW_EVIDENCE'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('SAMPLING')
] = 0.8
## the agent expects to see the LOW_REW_EVIDENCE observation with 20% probability, 
##   if the GAME_STATE is HIGH_REW, and the agent is in the SAMPLING state
_Aᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₁'].index('LOW_REW_EVIDENCE'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('SAMPLING')
] = 0.2

## the agent expects to see the LOW_REW_EVIDENCE observation with 80% probability, 
##   if the GAME_STATE is LOW_REW, and the agent is in the SAMPLING state
_Aᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₁'].index('LOW_REW_EVIDENCE'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('SAMPLING')
] = 0.8
## the agent expects to see the HIGH_REW_EVIDENCE observation with 20% probability, 
##   if the GAME_STATE is LOW_REW, and the agent is in the SAMPLING state
_Aᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₁'].index('HIGH_REW_EVIDENCE'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('SAMPLING')
] = 0.2

## quick way to do this
## Aᴳᵃᵐ[0][:, :, 0] = 1.0
## Aᴳᵃᵐ[0][:, :, 1] = 1.0
## Aᴳᵃᵐ[0][:, :, 2] = np.array([[0.8, 0.2], [0.2, 0.8], [0.0, 0.0]])

_Aᴾˡʳᴳᵃᵐ[0]

array([[[0. , 0. , 0.8],
        [0. , 0. , 0.2]],

       [[0. , 0. , 0.2],
        [0. , 0. , 0.8]],

       [[1. , 1. , 0. ],
        [1. , 1. , 0. ]]])
Set up the second modality’s likelihood mapping, correspond to how 
 (GAME_OUTCOME) are related to hidden states.

## regardless of the game state, if you're at the START, you see the 'neutral' outcome
_Aᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('NEUTRAL'), 
    :, 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('START')
] = 1.0

## regardless of the game state, if you're in the SAMPLING state, you see the 'neutral' outcome
_Aᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('NEUTRAL'), 
    :, 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('SAMPLING')
] = 1.0

## this is the distribution that maps from the "GAME_STATE" to the "GAME_OUTCOME" 
##   observation , in the case that "GAME_STATE" is `HIGH_REW`
_HIGH_REW_MAPPING_PLR_GAM = softmax(np.array([1.0, 0])) 

## this is the distribution that maps from the "GAME_STATE" to the "GAME_OUTCOME" 
##   observation , in the case that "GAME_STATE" is `LOW_REW`
_LOW_REW_MAPPING_PLR_GAM = softmax(np.array([0.0, 1.0]))

## fill out the A matrix using the reward probabilities
_Aᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('REWARD'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('PLAYING')
] = _HIGH_REW_MAPPING_PLR_GAM[0]
_Aᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('PUN'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('PLAYING')
] = _HIGH_REW_MAPPING_PLR_GAM[1]
_Aᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('REWARD'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('PLAYING')
] = _LOW_REW_MAPPING_PLR_GAM[0]
_Aᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('PUN'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'), 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('PLAYING')
] = _LOW_REW_MAPPING_PLR_GAM[1]

## quick way to do this
## Aᴳᵃᵐ[1][2, :, 0] = np.ones(num_states[0])
## Aᴳᵃᵐ[1][0:2, :, 1] = softmax(np.eye(num_obs[1] - 1)) # relationship of game state to reward observations (mapping between reward-state (first hidden state factor) and rewards (Good vs Bad))
## Aᴳᵃᵐ[1][2, :, 2] = np.ones(num_states[0])

_Aᴾˡʳᴳᵃᵐ[1]

array([[[0.        , 0.73105858, 0.        ],
        [0.        , 0.26894142, 0.        ]],

       [[0.        , 0.26894142, 0.        ],
        [0.        , 0.73105858, 0.        ]],

       [[1.        , 0.        , 1.        ],
        [1.        , 0.        , 1.        ]]])
Set up the third modality’s likelihood mapping, correspond to how 
 (ACTION_SELF_OBS) are related to hidden states.

_Aᴾˡʳᴳᵃᵐ[2][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₃'].index('START_O'), 
    :, 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('START')
] = 1.0
_Aᴾˡʳᴳᵃᵐ[2][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₃'].index('PLAY_O'), 
    :, 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('PLAYING')
] = 1.0
_Aᴾˡʳᴳᵃᵐ[2][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₃'].index('SAMPLE_O'), 
    :, 
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('SAMPLING')
] = 1.0

## quick way to do this
## modality_idx, factor_idx = 2, 2
## for sampling_state_i in num_states[factor_idx]:
##     Aᴳᵃᵐ[modality_idx][sampling_state_i,:,sampling_state_i] = 1.0

_Aᴾˡʳᴳᵃᵐ[2]

array([[[1., 0., 0.],
        [1., 0., 0.]],

       [[0., 1., 0.],
        [0., 1., 0.]],

       [[0., 0., 1.],
        [0., 0., 1.]]])
print(f'=== _sGam_car:\n{_sGam_car}')
print(f'=== _oGam_car:\n{_oGam_car}')
_Aᴾˡʳᴳᵃᵐ

=== _sGam_car:
[2, 3]
=== _oGam_car:
[3, 3, 3]
array([array([[[0. , 0. , 0.8],
               [0. , 0. , 0.2]],

              [[0. , 0. , 0.2],
               [0. , 0. , 0.8]],

              [[1. , 1. , 0. ],
               [1. , 1. , 0. ]]]),
       array([[[0.        , 0.73105858, 0.        ],
               [0.        , 0.26894142, 0.        ]],

              [[0.        , 0.26894142, 0.        ],
               [0.        , 0.73105858, 0.        ]],

              [[1.        , 0.        , 1.        ],
               [1.        , 0.        , 1.        ]]]),
       array([[[1., 0., 0.],
               [1., 0., 0.]],

              [[0., 1., 0.],
               [0., 1., 0.]],

              [[0., 0., 1.],
               [0., 0., 1.]]])], dtype=object)
## this is the (non-trivial) controllable factor, where there will be a >1-dimensional 
##   control state along this factor
_control_fac_idx_PlrGam = [1] ##.used in Agent constructor

_Bᴾˡʳᴳᵃᵐ = utils.obj_array(_sGam_num); print(f'{_sGam_num=}') ##.
print(f'{len(_Bᴾˡʳᴳᵃᵐ)=}')
_Bᴾˡʳᴳᵃᵐ

_sGam_num=2
len(_Bᴾˡʳᴳᵃᵐ)=2
array([None, None], dtype=object)
_Bᴾˡʳᴳᵃᵐ[0] = np.zeros((_sGam_car[0], _sGam_car[0], _uGam_car[0])); print(f'{_sGam_car[0]=}, {_sGam_car[0]=}, {_uGam_car[0]=}') ##.
_Bᴾˡʳᴳᵃᵐ[0]

_sGam_car[0]=2, _sGam_car[0]=2, _uGam_car[0]=1
array([[[0.],
        [0.]],

       [[0.],
        [0.]]])
_p_stochPlrGam = 0.0

## we cannot influence factor zero, set up the 'default' stationary dynamics - 
## one state just maps to itself at the next timestep with very high probability, 
## by default. So this means the reward state can change from one to another with 
## some low probability (p_stoch)

_Bᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'), 
    _labPlrGam['u']['uᴳᵃᵐ₁'].index('NULL_ACTION')
] = 1.0 - _p_stochPlrGam
_Bᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'), 
    _labPlrGam['u']['uᴳᵃᵐ₁'].index('NULL_ACTION')
] = _p_stochPlrGam

_Bᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'), 
    _labPlrGam['u']['uᴳᵃᵐ₁'].index('NULL_ACTION')
] = 1.0 - _p_stochPlrGam
_Bᴾˡʳᴳᵃᵐ[0][ ##.
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('HIGH_REW'),
    _labPlrGam['s']['sᴳᵃᵐ₁'].index('LOW_REW'), 
    _labPlrGam['u']['uᴳᵃᵐ₁'].index('NULL_ACTION')
] = _p_stochPlrGam

_Bᴾˡʳᴳᵃᵐ[0]

array([[[1.],
        [0.]],

       [[0.],
        [1.]]])
## setup our controllable factor
_Bᴾˡʳᴳᵃᵐ[1] = np.zeros((_sGam_car[1], _sGam_car[1], _uGam_car[1])) ##.
_Bᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('START'), 
    :, 
    _labPlrGam['u']['uᴳᵃᵐ₂'].index('START_ACTION')
] = 1.0
_Bᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('PLAYING'), 
    :, 
    _labPlrGam['u']['uᴳᵃᵐ₂'].index('PLAY_ACTION')
] = 1.0
_Bᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['s']['sᴳᵃᵐ₂'].index('SAMPLING'), 
    :, 
    _labPlrGam['u']['uᴳᵃᵐ₂'].index('SAMPLE_ACTION')
] = 1.0

_Bᴾˡʳᴳᵃᵐ[1]

array([[[1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.]],

       [[0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.]],

       [[0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]]])
print(f'=== _uGam_car:\n{_uGam_car}')
print(f'=== _sGam_car:\n{_sGam_car}')
_Bᴾˡʳᴳᵃᵐ

=== _uGam_car:
[1, 3]
=== _sGam_car:
[2, 3]
array([array([[[1.],
               [0.]],

              [[0.],
               [1.]]]), array([[[1., 0., 0.],
                                [1., 0., 0.],
                                [1., 0., 0.]],

                               [[0., 1., 0.],
                                [0., 1., 0.],
                                [0., 1., 0.]],

                               [[0., 0., 1.],
                                [0., 0., 1.],
                                [0., 0., 1.]]])], dtype=object)
Prior preferences
Now we parameterise the C vector, or the prior beliefs about observations. This will be used in the expression of the prior over actions, which is technically a softmax function of the negative expected free energy of each action. It is the equivalent of the exponentiated reward function in reinforcement learning treatments.

_Cᴾˡʳᴳᵃᵐ = utils.obj_array_zeros([o_car for o_car in _oGam_car]) ##.
_Cᴾˡʳᴳᵃᵐ

array([array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.])],
      dtype=object)
## make the observation we've a priori named `REWARD` actually desirable, by building 
##   a high prior expectation of encountering it 
_Cᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('REWARD'),
] = 1.0
## make the observation we've a prior named `PUN` actually aversive, by building a 
##   low prior expectation of encountering it
_Cᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('PUN'),
] = -1.0

## the above code implies the following for the `neutral' observation:
## we don't need to write this - but it's basically just saying that observing `NEUTRAL` 
##   is in between reward and punishment
_Cᴾˡʳᴳᵃᵐ[1][ ##.
    _labPlrGam['o']['oᴳᵃᵐ₂'].index('NEUTRAL'),
] = 0.0

_Cᴾˡʳᴳᵃᵐ[1]

array([ 1., -1.,  0.])
Initialise an instance of the Agent() class:
All you have to do is call Agent(generative_model_params...) where generative_model_params are your A, B, C’s… and whatever parameters of the generative model you want to specify

_agtPlrGam = Agent(
    A=_Aᴾˡʳᴳᵃᵐ, 
    B=_Bᴾˡʳᴳᵃᵐ, 
    C=_Cᴾˡʳᴳᵃᵐ, 
    control_fac_idx=_control_fac_idx_PlrGam
)
_agtPlrGam

<pymdp.agent.Agent at 0x7f10284e7520>
2.2 Game environment
This is the agent’s generative process for the game environment which embodies the system-under-steer for the player/game interaction.

Important note how the generative process doesn’t have to be described by A and B matrices - can just be the arbitrary ‘rules of the game’ that you ‘write in’ as a modeller. But here we just use the same transition/likelihood matrices to make the sampling process straightforward.

## transition/observation matrices characterising the generative process
_Ăᴳᵃᵐ = copy.deepcopy(_Aᴾˡʳᴳᵃᵐ) ##.

## True next-state will be calculated without a B matrix
_B̆ᴳᵃᵐ = copy.deepcopy(_Bᴾˡʳᴳᵃᵐ) ##.

3 Game/Config interaction
3.1 Game agent
This is the agent’s generative model for the config environment which embodies the system-under-steer for the game/config interaction.

3.1.1 State factors
We want to keep the Config environment really simple, just to expand to a multi-agent framework as easily as possible.

We will only have a single hidden state factor:

 (GAME_STATE_INVERSION)
3.1.1.1 
 (GAME_STATE_INVERSION)
The only factor is a binary variable representing whether the value of the Game’s GAME_STATE (represented by 
) should be inverted or not. It has two possible values or levels:

NO_GAME_STATE_INVERSION
, keep the value of 
 as it is
GAME_STATE_INVERSION
, invert the value of 
3.1.2 Observation modalities
The observation modalities consists of a single modality.

3.1.2.1 
 (Observations of the game state, GAME_STATE_INVERSION_OBS)
The only observation modality is the 
 (GAME_STATE_INVERSION_OBS) modality, and corresponds to observations that give the agent information about the GAME_STATE_INVERSION 
. There are two possible outcomes within this modality:

NO_REW_INVERSION_EVIDENCE
REW_INVERSION_EVIDENCE
So the first outcome can be described as lending evidence to the idea that the GAME_STATE_INVERSION 
 is not effective; the second outcome can be described as lending evidence to the idea that the GAME_STATE_INVERSION 
 is effective.

3.1.3 Control factors
The ‘control state’ factors are the agent’s representation of the control states (or actions) that it believes can influence the dynamics of the hidden states - i.e. hidden state factors that are under the influence of control states are are ‘controllable’. In practice, we often encode every hidden state factor as being under the influence of control states, but the ‘uncontrollable’ hidden state factors are driven by a trivially-1-dimensional control state or action-affordance. This trivial action simply ‘maintains the default environmental dynamics as they are’ i.e. does nothing. This will become more clear when we set up the transition model (the B matrices) below.

3.1.3.1 
 (REWARD_INVERSION_CONTROL)
The Game agent has the ability to attempt to steer the Config entity to try and maintain no inversion of the Game’s GAME_REWARD state. It does this by emitting an action signal on the Config entity:

NO_REW_INVERSION
REW_INVERSION
_labGamCfg = { ## labels for Game/Config (agent/environment) interaction
    ## agt
    "u": {
        "uᶜᶠᵍ₁": [ ## "REWARD_INVERSION_CONTROL" ## no cap C avail for superscript
            "NO_REW_INVERSION",
            "REW_INVERSION"
        ],
    },
    "s": {
        "sᶜᶠᵍ₁": [ ## "GAME_STATE_INVERSION"
            "NO_GAME_STATE_INVERSION", 
            "GAME_STATE_INVERSION"
        ],
    },
    "o": {
        "oᶜᶠᵍ₁": [ ## "GAME_STATE_INVERSION_OBS"
            "NO_REW_INVERSION_EVIDENCE",
            "REW_INVERSION_EVIDENCE",
        ],
    },
    ## env/sus
    "a": { 
        "aᶜᶠᵍ₁": [ ## "REWARD_INVERSION_CONTROL" ## no cap C avail
            "NO_REW_INVERSION",
            "REW_INVERSION"
        ],
    },
    "s̆": {
        "s̆ᶜᶠᵍ₁": [ ## "GAME_STATE_INVERSION"
            "NO_GAME_STATE_INVERSION", 
            "GAME_STATE_INVERSION"
        ],
    },    
    "y": {
        "yᶜᶠᵍ₁": [ ## "GAME_STATE_INVERSION_OBS"
            "NO_REW_INVERSION_EVIDENCE",
            "REW_INVERSION_EVIDENCE",
        ],
    },  
}
_oCfg_car,_oCfg_num, _sCfg_car,_sCfg_num, _uCfg_car,_uCfg_num = get_model_dimensions_from_labels(_labGamCfg) ##.
_oCfg_car,_oCfg_num, _sCfg_car,_sCfg_num, _uCfg_car,_uCfg_num

([2], 1, [2], 1, [2], 1)
print(f'{_uCfg_car=}') ##.cardinality of control factors
print(f'{_uCfg_num=}') ##.number of control factors

print(f'{_sCfg_car=}') ##.cardinality of state factors
print(f'{_sCfg_num=}') ##.number of state factors

print(f'{_oCfg_car=}') ##.cardinality of observation modalities
print(f'{_oCfg_num=}') ##.number of observation modalities

_uCfg_car=[2]
_uCfg_num=1
_sCfg_car=[2]
_sCfg_num=1
_oCfg_car=[2]
_oCfg_num=1
##.
_uCfg_fac_names = list(_labGamCfg['u'].keys()); print(f'{_uCfg_fac_names=}') ##.control factor names
_sCfg_fac_names = list(_labGamCfg['s'].keys()); print(f'{_sCfg_fac_names=}') ##.state factor names
_oCfg_mod_names = list(_labGamCfg['o'].keys()); print(f'{_oCfg_mod_names=}') ##.observation modality names

_uCfg_fac_names=['uᶜᶠᵍ₁']
_sCfg_fac_names=['sᶜᶠᵍ₁']
_oCfg_mod_names=['oᶜᶠᵍ₁']
##.
_aCfg_fac_names = list(_labGamCfg['a'].keys()); print(f'{_aCfg_fac_names=}') ##.control factor names
_s̆Cfg_fac_names = list(_labGamCfg['s̆'].keys()); print(f'{_s̆Cfg_fac_names=}') ##.state factor names
_yCfg_mod_names = list(_labGamCfg['y'].keys()); print(f'{_yCfg_mod_names=}') ##.observation modality names

_aCfg_fac_names=['aᶜᶠᵍ₁']
_s̆Cfg_fac_names=['s̆ᶜᶠᵍ₁']
_yCfg_mod_names=['yᶜᶠᵍ₁']
Setting up observation likelihood matrix - first main component of generative model

_Aᴳᵃᵐᶜᶠᵍ = utils.obj_array_zeros([[o_car] + _sCfg_car for o_car in _oCfg_car]) ##.
print(f'{len(_Aᴳᵃᵐᶜᶠᵍ)=}')
_Aᴳᵃᵐᶜᶠᵍ

len(_Aᴳᵃᵐᶜᶠᵍ)=1
array([array([[0., 0.],
              [0., 0.]])], dtype=object)
Set up the first modality’s likelihood mapping, correspond to how 
 (GAME_STATE_INVERSION_OBS) are related to hidden states.

## quick way to do this
_Aᴳᵃᵐᶜᶠᵍ[0][0, 0] = 1.0
_Aᴳᵃᵐᶜᶠᵍ[0][1, 1] = 1.0

_Aᴳᵃᵐᶜᶠᵍ[0]

array([[1., 0.],
       [0., 1.]])
print(f'=== _sCfg_car:\n{_sCfg_car}')
print(f'=== _oCfg_car:\n{_oCfg_car}')
_Aᴳᵃᵐᶜᶠᵍ

=== _sCfg_car:
[2]
=== _oCfg_car:
[2]
array([array([[1., 0.],
              [0., 1.]])], dtype=object)
## this is the (non-trivial) controllable factor, where there will be a >1-dimensional 
##   control state along this factor
## _control_fac_idx_GamCfg = [1] ##.used in Agent constructor

_Bᴳᵃᵐᶜᶠᵍ = utils.obj_array(_sCfg_num); print(f'{_sCfg_num=}') ##.
print(f'{len(_Bᴳᵃᵐᶜᶠᵍ)=}')
# _Bᴳᵃᵐᶜᶠᵍ = utils.obj_array([2, 2, 2]) ##.

# _Bᴳᵃᵐᶜᶠᵍ = np.zeros((2, 2, 2))  # Initialize a 2x2x2 matrix

_Bᴳᵃᵐᶜᶠᵍ

_sCfg_num=1
len(_Bᴳᵃᵐᶜᶠᵍ)=1
array([None], dtype=object)
_Bᴳᵃᵐᶜᶠᵍ[0] = np.zeros((_sCfg_car[0], _sCfg_car[0], _uCfg_car[0])); print(f'{_sCfg_car[0]=}, {_sCfg_car[0]=}, {_uCfg_car[0]=}') ##.
_Bᴳᵃᵐᶜᶠᵍ[0]

_sCfg_car[0]=2, _sCfg_car[0]=2, _uCfg_car[0]=2
array([[[0., 0.],
        [0., 0.]],

       [[0., 0.],
        [0., 0.]]])
_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('NO_REW_INVERSION')
] = 0.5
_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('NO_REW_INVERSION')
] = 0.5
_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('NO_REW_INVERSION')
] = 0.5
_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('NO_REW_INVERSION')
] = 0.5

_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('REW_INVERSION')
] = 0.5
_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('REW_INVERSION')
] = 0.5
_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('REW_INVERSION')
] = 0.5
_Bᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'),
    _labGamCfg['s']['sᶜᶠᵍ₁'].index('GAME_STATE_INVERSION'), 
    _labGamCfg['u']['uᶜᶠᵍ₁'].index('REW_INVERSION')
] = 0.5

_Bᴳᵃᵐᶜᶠᵍ[0]

array([[[0.5, 0.5],
        [0.5, 0.5]],

       [[0.5, 0.5],
        [0.5, 0.5]]])
print(f'=== _uCfg_car:\n{_uCfg_car}')
print(f'=== _sCfg_car:\n{_sCfg_car}')
_Bᴳᵃᵐᶜᶠᵍ

=== _uCfg_car:
[2]
=== _sCfg_car:
[2]
array([array([[[0.5, 0.5],
               [0.5, 0.5]],

              [[0.5, 0.5],
               [0.5, 0.5]]])], dtype=object)
Prior preferences
Now we parameterise the C vector, or the prior beliefs about observations. This will be used in the expression of the prior over actions, which is technically a softmax function of the negative expected free energy of each action. It is the equivalent of the exponentiated reward function in reinforcement learning treatments.

_Cᴳᵃᵐᶜᶠᵍ = utils.obj_array_zeros([o_car for o_car in _oCfg_car]) ##.
_Cᴳᵃᵐᶜᶠᵍ

array([array([0., 0.])], dtype=object)
_Cᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['o']['oᶜᶠᵍ₁'].index('NO_REW_INVERSION_EVIDENCE'),
] = 1.0

_Cᴳᵃᵐᶜᶠᵍ[0][ ##.
    _labGamCfg['o']['oᶜᶠᵍ₁'].index('REW_INVERSION_EVIDENCE'),
] = -1.0

_Cᴳᵃᵐᶜᶠᵍ[0]

array([ 1., -1.])
Initialise an instance of the Agent() class:
All you have to do is call Agent(generative_model_params...) where generative_model_params are your A, B, C’s… and whatever parameters of the generative model you want to specify

_agtGamCfg = Agent(
    A=_Aᴳᵃᵐᶜᶠᵍ, 
    B=_Bᴳᵃᵐᶜᶠᵍ, 
    C=_Cᴳᵃᵐᶜᶠᵍ, 
    ## control_fac_idx=_control_fac_idx_GamCfg; gives error
)
_agtGamCfg

<pymdp.agent.Agent at 0x7f0fc00d0490>
3.2 Config environment
This is the agent’s generative process for the game environment which embodies the system-under-steer for the game/config interaction.

Important note how the generative process doesn’t have to be described by A and B matrices - can just be the arbitrary ‘rules of the game’ that you ‘write in’ as a modeller. But here we just use the same transition/likelihood matrices to make the sampling process straightforward.

## transition/observation matrices characterising the generative process
_Ăᶜᶠᵍ = copy.deepcopy(_Aᴳᵃᵐᶜᶠᵍ)
_B̆ᶜᶠᵍ = copy.deepcopy(_Bᴳᵃᵐᶜᶠᵍ)

4 Run simulation
Initialise the simulation

_T = 20 ## number of timesteps in the simulation

## Player/Game
_s̆ᴳᵃᵐ = [ ## initial (true) state
    _labPlrGam['s̆']['s̆ᴳᵃᵐ₁'].index('HIGH_REW'), 
    _labPlrGam['s̆']['s̆ᴳᵃᵐ₂'].index('START')
]; print(f'{_s̆ᴳᵃᵐ=}')

_yᴳᵃᵐ = [ ## initial observation
    _labPlrGam['y']['yᴳᵃᵐ₁'].index('NO_EVIDENCE'), 
    _labPlrGam['y']['yᴳᵃᵐ₂'].index('NEUTRAL'),
    _labPlrGam['y']['yᴳᵃᵐ₃'].index('START_O')
]; print(f'{_yᴳᵃᵐ=}')

_s̆ᴳᵃᵐ=[0, 0]
_yᴳᵃᵐ=[2, 2, 0]
## Game/Config
_s̆ᶜᶠᵍ = [ ## initial (true) state
    _labGamCfg['s̆']['s̆ᶜᶠᵍ₁'].index('NO_GAME_STATE_INVERSION')
]; print(f'{_s̆ᶜᶠᵍ=}')

_yᶜᶠᵍ = [ ## initial observation
    _labGamCfg['y']['yᶜᶠᵍ₁'].index('NO_REW_INVERSION_EVIDENCE')
]; print(f'{_yᶜᶠᵍ=}')

_s̆ᶜᶠᵍ=[0]
_yᶜᶠᵍ=[0]
Create some string names for the state, observation, and action indices to help with print statements

## Player/Game
_uGam_val_names = [_labPlrGam['u'][cfn] for cfn in _uGam_fac_names]; print(f'{_uGam_val_names=}') ##.
_sGam_val_names = [_labPlrGam['s'][sfn] for sfn in _sGam_fac_names]; print(f'{_sGam_val_names=}') ##.
_oGam_val_names = [_labPlrGam['o'][omn] for omn in _oGam_mod_names]; print(f'{_oGam_val_names=}')

_aGam_val_names = [_labPlrGam['a'][cfn] for cfn in _aGam_fac_names]; print(f'{_aGam_val_names=}') ##.
_s̆Gam_val_names = [_labPlrGam['s̆'][sfn] for sfn in _s̆Gam_fac_names]; print(f'{_s̆Gam_val_names=}') ##.
_yGam_val_names = [_labPlrGam['y'][omn] for omn in _yGam_mod_names]; print(f'{_yGam_val_names=}')

_uGam_val_names=[['NULL_ACTION'], ['START_ACTION', 'PLAY_ACTION', 'SAMPLE_ACTION']]
_sGam_val_names=[['HIGH_REW', 'LOW_REW'], ['START', 'PLAYING', 'SAMPLING']]
_oGam_val_names=[['HIGH_REW_EVIDENCE', 'LOW_REW_EVIDENCE', 'NO_EVIDENCE'], ['REWARD', 'PUN', 'NEUTRAL'], ['START_O', 'PLAY_O', 'SAMPLE_O']]
_aGam_val_names=[['NULL_ACTION'], ['START_ACTION', 'PLAY_ACTION', 'SAMPLE_ACTION']]
_s̆Gam_val_names=[['HIGH_REW', 'LOW_REW'], ['START', 'PLAYING', 'SAMPLING']]
_yGam_val_names=[['HIGH_REW_EVIDENCE', 'LOW_REW_EVIDENCE', 'NO_EVIDENCE'], ['REWARD', 'PUN', 'NEUTRAL'], ['START_O', 'PLAY_O', 'SAMPLE_O']]
## Game/Config
_uCfg_val_names = [_labGamCfg['u'][cfn] for cfn in _uCfg_fac_names]; print(f'{_uCfg_val_names=}') ##.
_sCfg_val_names = [_labGamCfg['s'][sfn] for sfn in _sCfg_fac_names]; print(f'{_sCfg_val_names=}') ##.
_oCfg_val_names = [_labGamCfg['o'][omn] for omn in _oCfg_mod_names]; print(f'{_oCfg_val_names=}')

_aCfg_val_names = [_labGamCfg['a'][cfn] for cfn in _aCfg_fac_names]; print(f'{_aCfg_val_names=}') ##.
_s̆Cfg_val_names = [_labGamCfg['s̆'][sfn] for sfn in _s̆Cfg_fac_names]; print(f'{_s̆Cfg_val_names=}') ##.
_yCfg_val_names = [_labGamCfg['y'][omn] for omn in _yCfg_mod_names]; print(f'{_yCfg_val_names=}')

_uCfg_val_names=[['NO_REW_INVERSION', 'REW_INVERSION']]
_sCfg_val_names=[['NO_GAME_STATE_INVERSION', 'GAME_STATE_INVERSION']]
_oCfg_val_names=[['NO_REW_INVERSION_EVIDENCE', 'REW_INVERSION_EVIDENCE']]
_aCfg_val_names=[['NO_REW_INVERSION', 'REW_INVERSION']]
_s̆Cfg_val_names=[['NO_GAME_STATE_INVERSION', 'GAME_STATE_INVERSION']]
_yCfg_val_names=[['NO_REW_INVERSION_EVIDENCE', 'REW_INVERSION_EVIDENCE']]
def act(agt, a_facs, a_fac_names, a_val_names, s̆_fac_names, t):
    if(t == 0): ##.at t=0 agent has no q_pi yet, so no .sample_action()
        action = np.array([0.0, 0.0]) ##.
        print(f"_a: {[(a_fac_names[a], a_val_names[a][int(action[a])]) for a in range(len(s̆_fac_names))]}")
    else: ## t > 0
        action = agt.sample_action()
        ## min_F.append(np.min(_agent.F)) ##.does not have .F
        print(f"_a: {[(a_fac_names[a], a_val_names[a][int(action[a])]) for a in range(len(s̆_fac_names))]}") ##.
        for afi, afn in enumerate(a_fac_names):
            a_facs[afn].append(a_val_names[afi][int(action[afi])])
    return action

def future(agt, qIpiIs, GNegs):
    ## _agent.infer_policies()
    qIpiI, GNeg = agt.infer_policies() ##.posterior over policies and negative EFE
    print(f'{qIpiI=}')
    print(f'{GNeg=}')
    qIpiIs.append(qIpiI)
    GNegs.append(GNeg)

def next(s̆_facs, action, s̆, B̆, s̆_fac_names, s̆_val_names):
    for sfi, sf in enumerate(s̆):
        s̆[sfi] = utils.sample(B̆[sfi][:, sf, int(action[sfi])]) ##.
    print(f"_s̆: {[(s̆_fac_names[sfi], s̆_val_names[sfi][s̆[sfi]]) for sfi in range(len(s̆_fac_names))]}") ##.
    for sfi, sfn in enumerate(s̆_fac_names):
        s̆_facs[sfn].append(s̆_val_names[sfi][s̆[sfi]])

def nextWithoutB(s̆_facs, s̆, s̆_fac_names, s̆_val_names, yCfg_mods):
    ## for sfi, sf in enumerate(s̆):
    ##     s̆[sfi] = utils.sample(B̆[sfi][:, sf, int(action[sfi])]) ##.
    print(f'!!! BEFORE: {s̆=}')
    if len(yCfg_mods['yᶜᶠᵍ₁']) > 0:
        yCfg = yCfg_mods['yᶜᶠᵍ₁'][-1]
        print(f"!!! {yCfg=}")
        if yCfg == "REW_INVERSION_EVIDENCE":
            ## flip value of s̆ᴳᵃᵐ₁
            s̆_0_new = (s̆[0] + 1) % 2
            # s̆_1_new = (s̆[1] + 1) % 2
            s̆_1_new = s̆[1]
            s̆ = [s̆_0_new, s̆_1_new]
        print(f'!!! AFTER: {s̆=}')
    print(f"_s̆: {[(s̆_fac_names[sfi], s̆_val_names[sfi][s̆[sfi]]) for sfi in range(len(s̆_fac_names))]}") ##.
    for sfi, sfn in enumerate(s̆_fac_names):
        s̆_facs[sfn].append(s̆_val_names[sfi][s̆[sfi]])

def observe(y_mods, y, Ă, s̆, _y_mod_names, _y_val_names):
    for omi, _ in enumerate(y): ##.
        if len(s̆) == 1:
            y[omi] = utils.sample(Ă[omi][:, s̆[0]]) ##.
        elif len(s̆) == 2:
            y[omi] = utils.sample(Ă[omi][:, s̆[0], s̆[1]]) ##.
        else:
            print(f'ERROR: {len(s̆)=} not handled!')
    print(f"_y: {[(_y_mod_names[omi], _y_val_names[omi][y[omi]]) for omi in range(len(_y_mod_names))]}") ##.
    for ymi, ymn in enumerate(_y_mod_names):
        y_mods[ymn].append(_y_val_names[ymi][y[ymi]])

def infer(agt, s_facs, y, s_fac_names, lab):
    belief_state = agt.infer_states(y) ##.
    print(f"Beliefs: {[(s_fac_names[sfi], belief_state[sfi].round(3).T) for sfi in range(len(s_fac_names))]}") ##.
    for sfi, sfn in enumerate(s_fac_names):
        s_facs[sfn].append( lab['s'][sfn][int(np.argmax(belief_state[sfi].round(3).T))] ) ##.
    ## exmpl: print(f"_s̆ᴳᵃᵐ: {[(_s̆_fac_names[sfi], _s̆_idx_names[sfi][_s̆ᴳᵃᵐ[sfi]]) for sfi in range(len(_s̆_fac_names))]}") ##.
    ## print(f"_sᴳᵃᵐ: {[(_sta_fac_names[sfi], _sta_idx_names[sfi]) for sfi in range(len(_sta_fac_names))]}") ##.    

_aPlrGam_facs = {'aᴳᵃᵐ₁': [], 'aᴳᵃᵐ₂': []}
_sPlrGam_facs = {'sᴳᵃᵐ₁': [], 'sᴳᵃᵐ₂': []}
_s̆Gam_facs = {'s̆ᴳᵃᵐ₁': [], 's̆ᴳᵃᵐ₂': []}
_yGam_mods = {'yᴳᵃᵐ₁': [], 'yᴳᵃᵐ₂': [], 'yᴳᵃᵐ₃': []}

_aGamCfg_facs = {'aᶜᶠᵍ₁': []}
_sGamCfg_facs = {'sᶜᶠᵍ₁': []}
_s̆Cfg_facs = {'s̆ᶜᶠᵍ₁': []}
_yCfg_mods = {'yᶜᶠᵍ₁': []}
## min_F = []
_qPlrGamIpiIs = []
_GPlrGamNegs = []

_qGamCfgIpiIs = []
_GGamCfgNegs = []

for t in range(_T):
    print(f"\nTime {t}:")

    ### act
    print('___act___')
    actionPlrGam = act(_agtPlrGam, _aPlrGam_facs, _aGam_fac_names, _aGam_val_names, _s̆Gam_fac_names, t)
    actionGamCfg = act(_agtGamCfg, _aGamCfg_facs, _aCfg_fac_names, _aCfg_val_names, _s̆Cfg_fac_names, t)

    ### future
    print('___future___')
    future(_agtPlrGam, _qPlrGamIpiIs, _GPlrGamNegs)
    future(_agtGamCfg, _qGamCfgIpiIs, _GGamCfgNegs)

    ### next
    print('___next___')
    ## next(_s̆Gam_facs, actionPlrGam, _s̆ᴳᵃᵐ, _B̆ᴳᵃᵐ, _s̆Gam_fac_names, _s̆Gam_val_names)
    nextWithoutB(_s̆Gam_facs, _s̆ᴳᵃᵐ, _s̆Gam_fac_names, _s̆Gam_val_names, _yCfg_mods)
    next(_s̆Cfg_facs, actionGamCfg, _s̆ᶜᶠᵍ, _B̆ᶜᶠᵍ, _s̆Cfg_fac_names, _s̆Cfg_val_names)

    ### observe
    print('___observe___')
    observe(_yGam_mods, _yᴳᵃᵐ, _Ăᴳᵃᵐ, _s̆ᴳᵃᵐ, _yGam_mod_names, _yGam_val_names)
    observe(_yCfg_mods, _yᶜᶠᵍ, _Ăᶜᶠᵍ, _s̆ᶜᶠᵍ, _yCfg_mod_names, _yCfg_val_names)

    ### infer
    print('___infer___')
    infer(_agtPlrGam, _sPlrGam_facs, _yᴳᵃᵐ, _sGam_fac_names, _labPlrGam)
    infer(_agtGamCfg, _sGamCfg_facs, _yᶜᶠᵍ, _sCfg_fac_names, _labGamCfg)


Time 0:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'START_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14510120e-04, 9.98770512e-01, 6.14977437e-04])
GNeg=array([-3.60483043, -3.1427395 , -3.60478292])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 1:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14346833e-04, 9.98771134e-01, 6.14519376e-04])
GNeg=array([-3.60483043, -3.14272285, -3.60481288])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 2:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14286774e-04, 9.98771362e-01, 6.14350939e-04])
GNeg=array([-3.60483043, -3.14271672, -3.6048239 ])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 3:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14264681e-04, 9.98771446e-01, 6.14288984e-04])
GNeg=array([-3.60483043, -3.14271447, -3.60482796])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 4:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14256554e-04, 9.98771477e-01, 6.14266193e-04])
GNeg=array([-3.60483043, -3.14271364, -3.60482945])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'PUN'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 5:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14264681e-04, 9.98771446e-01, 6.14288984e-04])
GNeg=array([-3.60483043, -3.14271447, -3.60482796])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 6:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14256554e-04, 9.98771477e-01, 6.14266193e-04])
GNeg=array([-3.60483043, -3.14271364, -3.60482945])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 7:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14253564e-04, 9.98771489e-01, 6.14257809e-04])
GNeg=array([-3.60483043, -3.14271334, -3.60483   ])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'PUN'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 8:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14256554e-04, 9.98771477e-01, 6.14266193e-04])
GNeg=array([-3.60483043, -3.14271364, -3.60482945])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 9:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14253564e-04, 9.98771489e-01, 6.14257809e-04])
GNeg=array([-3.60483043, -3.14271334, -3.60483   ])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 10:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14252464e-04, 9.98771493e-01, 6.14254725e-04])
GNeg=array([-3.60483043, -3.14271322, -3.6048302 ])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 11:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14252296e-04, 9.98771494e-01, 6.14253402e-04])
GNeg=array([-3.60483038, -3.14271316, -3.60483026])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 12:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251998e-04, 9.98771495e-01, 6.14253104e-04])
GNeg=array([-3.60483041, -3.14271316, -3.6048303 ])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 13:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251888e-04, 9.98771495e-01, 6.14252994e-04])
GNeg=array([-3.60483042, -3.14271316, -3.60483031])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 14:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251848e-04, 9.98771495e-01, 6.14252954e-04])
GNeg=array([-3.60483043, -3.14271316, -3.60483031])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 15:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251833e-04, 9.98771495e-01, 6.14252939e-04])
GNeg=array([-3.60483043, -3.14271316, -3.60483032])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 16:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251827e-04, 9.98771495e-01, 6.14252933e-04])
GNeg=array([-3.60483043, -3.14271316, -3.60483032])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'PUN'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 17:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251833e-04, 9.98771495e-01, 6.14252939e-04])
GNeg=array([-3.60483043, -3.14271316, -3.60483032])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'NO_GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'PUN'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'NO_REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([1., 0.]))]

Time 18:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251848e-04, 9.98771495e-01, 6.14252954e-04])
GNeg=array([-3.60483043, -3.14271316, -3.60483031])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='NO_REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(0), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'HIGH_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'PUN'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]

Time 19:
___act___
_a: [('aᴳᵃᵐ₁', 'NULL_ACTION'), ('aᴳᵃᵐ₂', 'PLAY_ACTION')]
_a: [('aᶜᶠᵍ₁', 'NO_REW_INVERSION')]
___future___
qIpiI=array([6.14251888e-04, 9.98771495e-01, 6.14252994e-04])
GNeg=array([-3.60483042, -3.14271316, -3.60483031])
qIpiI=array([0.5, 0.5])
GNeg=array([-0.43378072, -0.43378072])
___next___
!!! BEFORE: s̆=[np.int64(0), np.int64(1)]
!!! yCfg='REW_INVERSION_EVIDENCE'
!!! AFTER: s̆=[np.int64(1), np.int64(1)]
_s̆: [('s̆ᴳᵃᵐ₁', 'LOW_REW'), ('s̆ᴳᵃᵐ₂', 'PLAYING')]
_s̆: [('s̆ᶜᶠᵍ₁', 'GAME_STATE_INVERSION')]
___observe___
_y: [('yᴳᵃᵐ₁', 'NO_EVIDENCE'), ('yᴳᵃᵐ₂', 'REWARD'), ('yᴳᵃᵐ₃', 'PLAY_O')]
_y: [('yᶜᶠᵍ₁', 'REW_INVERSION_EVIDENCE')]
___infer___
Beliefs: [('sᴳᵃᵐ₁', array([1., 0.])), ('sᴳᵃᵐ₂', array([0., 1., 0.]))]
Beliefs: [('sᶜᶠᵍ₁', array([0., 1.]))]
colors = [
{'NULL_ACTION':'black'}, ## aGam_1
{'START_ACTION':'red', 'PLAY_ACTION':'green', 'SAMPLE_ACTION': 'blue'}, ## aGam_2

{'HIGH_REW':'orange', 'LOW_REW':'purple'}, ## sGam_1
{'START':'red', 'PLAYING':'green', 'SAMPLING': 'blue'}, ## sGam_2

{'HIGH_REW':'orange', 'LOW_REW':'purple'}, ## qIsIGam_1
{'START':'red', 'PLAYING':'green', 'SAMPLING': 'blue'}, ## qIsIGam_2

{'HIGH_REW_EVIDENCE':'orange', 'LOW_REW_EVIDENCE':'purple', 'NO_EVIDENCE':'pink'}, ## yGam_1
{'REWARD':'red', 'PUN':'green', 'NEUTRAL': 'blue'}, ## yGam_2
{'START_O':'red', 'PLAY_O':'green', 'SAMPLE_O': 'blue'} ## yGam_3
]

ylabel_size = 12
msi = 7 ## markersize for Line2D, diameter in points
siz = (msi/2)**2 * np.pi ## size for scatter, area of marker in points squared

fig = plt.figure(figsize=(9, 6))
## gs = GridSpec(6, 1, figure=fig, height_ratios=[1, 3, 1, 3, 3, 1])
gs = GridSpec(9, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1])
ax = [fig.add_subplot(gs[i]) for i in range(9)]

i = 0
ax[i].set_title(f'Player/Game interaction', fontweight='bold',fontsize=14)
y_pos = 0
for t, s in zip(range(_T), _aPlrGam_facs['aᴳᵃᵐ₁']): ## 'NULL'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz)
ax[i].set_ylabel('$a^{\mathrm{Gam}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
ax[i].set_xticklabels([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor='black',markersize=msi,label='NULL_ACTION')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 1
y_pos = 0
for t, s in zip(range(_T), _aPlrGam_facs['aᴳᵃᵐ₂']): ## 'PLAYING_VS_SAMPLING_CONTROL'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz, label=s)
ax[i].set_ylabel('$a^{\mathrm{Gam}}_{2t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['START_ACTION'],markersize=msi,label='START_ACTION'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['PLAY_ACTION'],markersize=msi,label='PLAY_ACTION'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['SAMPLE_ACTION'],markersize=msi,label='SAMPLE_ACTION')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 2
y_pos = 0
for t, s in zip(range(_T), _sPlrGam_facs['sᴳᵃᵐ₁']): ## 'GAME_STATE'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz)
ax[i].set_ylabel('$s^{\mathrm{Gam}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
ax[i].set_xticklabels([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['HIGH_REW'],markersize=msi,label='HIGH_REW'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['LOW_REW'],markersize=msi,label='LOW_REW')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 3
y_pos = 0
for t, s in zip(range(_T), _sPlrGam_facs['sᴳᵃᵐ₂']): ## 'PLAYING_VS_SAMPLING'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz, label=s)
ax[i].set_ylabel('$s^{\mathrm{Gam}}_{2t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['START'],markersize=msi,label='START'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['PLAYING'],markersize=msi,label='PLAYING'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['SAMPLING'],markersize=msi,label='SAMPLING')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 4
y_pos = 0
for t, s in zip(range(_T), _sPlrGam_facs['sᴳᵃᵐ₁']): ## 'GAME_STATE'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz)
ax[i].set_ylabel('$q(s)^{\mathrm{Gam}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
ax[i].set_xticklabels([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['HIGH_REW'],markersize=msi,label='HIGH_REW'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['LOW_REW'],markersize=msi,label='LOW_REW')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 5
y_pos = 0
for t, s in zip(range(_T), _sPlrGam_facs['sᴳᵃᵐ₂']): ## 'PLAYING_VS_SAMPLING'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz, label=s)
ax[i].set_ylabel('$q(s)^{\mathrm{Gam}}_{2t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['START'],markersize=msi,label='START'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['PLAYING'],markersize=msi,label='PLAYING'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['SAMPLING'],markersize=msi,label='SAMPLING')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 6
y_pos = 0
for t, s in zip(range(_T), _yGam_mods['yᴳᵃᵐ₁']): ## 'GAME_STATE_OBS'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz, label=s)
ax[i].set_ylabel('$y^{\mathrm{Gam}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['HIGH_REW_EVIDENCE'],markersize=msi,label='HIGH_REW_EVIDENCE'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['LOW_REW_EVIDENCE'],markersize=msi,label='LOW_REW_EVIDENCE'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['NO_EVIDENCE'],markersize=msi,label='NO_EVIDENCE')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 7
y_pos = 0
for t, s in zip(range(_T), _yGam_mods['yᴳᵃᵐ₂']): ## 'GAME_OUTCOME'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz, label=s)
ax[i].set_ylabel('$y^{\mathrm{Gam}}_{2t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['REWARD'],markersize=msi,label='REWARD'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['PUN'],markersize=msi,label='PUN'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['NEUTRAL'],markersize=msi,label='NEUTRAL')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 8
y_pos = 0
for t, s in zip(range(_T), _yGam_mods['yᴳᵃᵐ₃']): ## 'ACTION_SELF_OBS'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz, label=s)
ax[i].set_ylabel('$y^{\mathrm{Gam}}_{3t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['START_O'],markersize=msi,label='START_O'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['PLAY_O'],markersize=msi,label='PLAY_O'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['SAMPLE_O'],markersize=msi,label='SAMPLE_O')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
## ax[i].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
ax[i].set_xlabel('$\mathrm{time,}\ t$', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1) ## Adjust this value as needed
plt.show()



colors = [
{'NO_REW_INVERSION':'black', 'REW_INVERSION':'red'}, ## aCfg_1

{'NO_GAME_STATE_INVERSION':'orange', 'GAME_STATE_INVERSION':'purple'}, ## sCfg_1

{'NO_GAME_STATE_INVERSION':'orange', 'GAME_STATE_INVERSION':'purple'}, ## qIsICfg_1

{'NO_REW_INVERSION_EVIDENCE':'orange', 'REW_INVERSION_EVIDENCE':'purple'}, ## yCfg_1
]

ylabel_size = 12
msi = 7 ## markersize for Line2D, diameter in points
siz = (msi/2)**2 * np.pi ## size for scatter, area of marker in points squared

fig = plt.figure(figsize=(9, 6))
## gs = GridSpec(6, 1, figure=fig, height_ratios=[1, 3, 1, 3, 3, 1])
gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1])
ax = [fig.add_subplot(gs[i]) for i in range(4)]

i = 0
ax[i].set_title(f'Game/Config interaction', fontweight='bold',fontsize=14)
y_pos = 0
for t, s in zip(range(_T), _aGamCfg_facs['aᶜᶠᵍ₁']):
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz)
ax[i].set_ylabel('$a^{\mathrm{Cfg}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
ax[i].set_xticklabels([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor='black',markersize=msi,label='NO_REW_INVERSION'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='red',markersize=msi,label='REW_INVERSION')
]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 1
y_pos = 0
for t, s in zip(range(_T), _sGamCfg_facs['sᶜᶠᵍ₁']):
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz)
ax[i].set_ylabel('$s^{\mathrm{Cfg}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
ax[i].set_xticklabels([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['NO_GAME_STATE_INVERSION'],markersize=msi,label='NO_GAME_STATE_INVERSION'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['GAME_STATE_INVERSION'],markersize=msi,label='GAME_STATE_INVERSION')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 2
y_pos = 0
for t, s in zip(range(_T), _sGamCfg_facs['sᶜᶠᵍ₁']):
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz)
ax[i].set_ylabel('$q(s)^{\mathrm{Cfg}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
ax[i].set_xticklabels([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['NO_GAME_STATE_INVERSION'],markersize=msi,label='NO_GAME_STATE_INVERSION'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['GAME_STATE_INVERSION'],markersize=msi,label='GAME_STATE_INVERSION')]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

i = 3
y_pos = 0
for t, s in zip(range(_T), _yCfg_mods['yᶜᶠᵍ₁']): ## 'GAME_STATE_OBS'
    ax[i].scatter(t, y_pos, color=colors[i][s], s=siz, label=s)
ax[i].set_ylabel('$y^{\mathrm{Cfg}}_{1t}$', rotation=0, fontweight='bold', fontsize=ylabel_size)
ax[i].set_yticks([])
leg_items = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['NO_REW_INVERSION_EVIDENCE'],markersize=msi,label='NO_REW_INVERSION_EVIDENCE'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i]['REW_INVERSION_EVIDENCE'],markersize=msi,label='REW_INVERSION_EVIDENCE'),
]
ax[i].legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, labelspacing=0.1)
ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)

ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
## ax[i].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
ax[i].set_xlabel('$\mathrm{time,}\ t$', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1) ## Adjust this value as needed
plt.show()

