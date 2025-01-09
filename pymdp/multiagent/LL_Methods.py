import os
import sys
from pathlib import Path
import logging
import numpy as np
import json
from datetime import datetime
import subprocess
import importlib
import matplotlib.pyplot as plt
import copy
import traceback

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup Python paths to include both local pymdp and parent project."""
    try:
        # Get the absolute path to the script's directory
        script_dir = Path(__file__).resolve().parent
        
        # Add parent directory (pymdp root) to Python path
        project_root = script_dir.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            logger.info(f"Added project root to Python path: {project_root}")
            
        # Add the multiagent directory to Python path
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
            logger.info(f"Added multiagent directory to Python path: {script_dir}")
            
        # Add the parent directory of project_root to Python path
        # This is needed to import pymdp as a package
        package_parent = project_root.parent
        if str(package_parent) not in sys.path:
            sys.path.insert(0, str(package_parent))
            logger.info(f"Added package parent to Python path: {package_parent}")
            
        # Log current Python path for debugging
        logger.debug("Current Python path:")
        for path in sys.path:
            logger.debug(f"  {path}")
            
        # Try importing pymdp to verify paths are correct
        import pymdp
        logger.info(f"Successfully imported pymdp from {pymdp.__file__}")
            
        return True
    except Exception as e:
        logger.error(f"Error setting up paths: {e}")
        logger.error(traceback.format_exc())
        return False

def check_and_install_dependencies():
    """Check and install required Python packages."""
    required_packages = {
        'numpy': 'numpy>=1.19.0',
        'matplotlib': 'matplotlib>=3.3.0',
        'scipy': 'scipy>=1.5.0',
        'torch': 'torch>=1.7.0',
        'pandas': 'pandas>=1.1.0',
        'seaborn': 'seaborn>=0.11.0'
    }
    
    for package, pip_name in required_packages.items():
        try:
            importlib.import_module(package)
            logger.info(f"Found {package}")
        except ImportError:
            logger.warning(f"{package} not found. Installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                return False
    return True

def check_pymdp_installation():
    """Check if pymdp modules are accessible."""
    try:
        # Get the absolute path to the script's directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        # Log the files we're looking for
        logger.debug(f"Looking for pymdp modules in {project_root}")
        expected_files = ['inference.py', 'control.py', 'utils.py', 'maths.py']
        for file in expected_files:
            file_path = project_root / file
            if file_path.exists():
                logger.debug(f"Found {file} at {file_path}")
            else:
                logger.warning(f"Missing {file} at {file_path}")
        
        # Try importing core modules directly
        from pymdp.utils import obj_array_zeros, sample
        logger.debug("Successfully imported utils module")
        
        from pymdp.maths import softmax
        logger.debug("Successfully imported maths module")
        
        from pymdp.inference import infer_states
        logger.debug("Successfully imported inference module")
        
        from pymdp.control import sample_action, infer_policies
        logger.debug("Successfully imported control module")
        
        logger.info("Successfully imported all pymdp modules")
        return True
    except ImportError as e:
        logger.error(f"Error importing pymdp modules: {e}")
        logger.error("Make sure you're in the correct directory.")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Python path: {sys.path}")
        return False

def import_dependencies():
    """Import all required dependencies."""
    try:
        # Absolute imports instead of relative
        from pymdp.utils import obj_array_zeros, sample
        from pymdp.maths import softmax
        from pymdp.inference import infer_states
        from pymdp.control import sample_action, infer_policies
        
        # Create Agent class
        global Agent
        class Agent:
            def __init__(self, A, B, C, control_fac_idx=None):
                """Initialize Active Inference agent.
                
                Args:
                    A (array): Observation model (likelihood)
                    B (array): Transition model (dynamics)
                    C (array): Prior preferences
                    control_fac_idx (list): Control factor indices
                """
                self.A = A  # Observation model (likelihood)
                self.B = B  # Transition model (dynamics)
                self.C = C  # Prior preferences
                self.control_fac_idx = control_fac_idx
                logger.debug(f"Initialized Agent with control_fac_idx={control_fac_idx}")
                
            def infer_states(self, obs):
                """Infer hidden states given observations."""
                logger.debug(f"Inferring states for observation shape: {np.array(obs).shape}")
                return infer_states(obs, self.A, self.B)
                
            def sample_action(self):
                """Sample action based on current beliefs and preferences."""
                logger.debug("Sampling action")
                return sample_action(self.A, self.B, self.C, self.control_fac_idx)
                
            def infer_policies(self):
                """Infer policies and calculate expected free energy."""
                logger.debug("Inferring policies")
                return infer_policies(self.A, self.B, self.C, self.control_fac_idx)
        
        logger.info("Successfully imported pymdp modules and created Agent class")
        return True
    except ImportError as e:
        logger.error(f"Error importing dependencies: {e}")
        logger.error("Make sure you're in the correct directory.")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Python path: {sys.path}")
        return False

def setup_output_directories():
    """Create output directories for simulation results."""
    try:
        current_dir = Path(os.getcwd())
        output_base = current_dir / 'Outputs'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_base / f'run_{timestamp}'
        analysis_dir = run_dir / 'analysis'
        figures_dir = run_dir / 'figures'

        # Create directories with error checking
        for dir_path in [output_base, run_dir, analysis_dir, figures_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                raise
        
        return timestamp, run_dir, analysis_dir, figures_dir
    except Exception as e:
        logger.error(f"Error setting up output directories: {e}")
        raise

def get_model_dimensions_from_labels(labels):
    """Get model dimensions from labels dictionary."""
    # Get cardinality and number of control factors
    a_car = [len(val) for val in labels['a'].values()]  # cardinality of control factors
    a_num = len(a_car)  # number of control factors

    # Get cardinality and number of state factors
    s_car = [len(val) for val in labels['s'].values()]  # cardinality of state factors
    s_num = len(s_car)  # number of state factors

    # Get cardinality and number of observation modalities
    y_car = [len(val) for val in labels['y'].values()]  # cardinality of observation modalities
    y_num = len(y_car)  # number of observation modalities

    return y_car, y_num, s_car, s_num, a_car, a_num

def run_simulation():
    """Run the Active Web Campaign simulation."""
    try:
        # Import local modules
        from pymdp.multiagent.LL_GM_Admin import (_labAdm, _Aᴬᵈᵐ, _Bᴬᵈᵐ, _Cᴬᵈᵐ, _B̆ᴬᵈᵐ, 
                                _Ăᴬᵈᵐ, _control_fac_idx_Adm)
        from pymdp.multiagent.LL_GM_Website import (_labWeb, _Aᵂᵉᵇ, _Bᵂᵉᵇ, _Cᵂᵉᵇ, 
                                  _B̆ᵂᵉᵇ, _Ăᵂᵉᵇ)
        
        _T = 25  # timesteps in the simulation

        # Initialize agents
        _agtAdm = Agent(A=_Aᴬᵈᵐ, B=_Bᴬᵈᵐ, C=_Cᴬᵈᵐ, control_fac_idx=_control_fac_idx_Adm)
        _agtWeb = Agent(A=_Aᵂᵉᵇ, B=_Bᵂᵉᵇ, C=_Cᵂᵉᵇ)

        # Define factor names
        _aAdm_fac_names = list(_labAdm['a'].keys())
        _sWeb_fac_names = list(_labAdm['s'].keys())
        _s̆Web_fac_names = list(_labAdm['s̆'].keys())
        _yWeb_mod_names = list(_labAdm['y'].keys())

        _aWeb_fac_names = list(_labWeb['a'].keys())
        _sCsr_fac_names = list(_labWeb['s'].keys())
        _s̆Csr_fac_names = list(_labWeb['s̆'].keys())
        _yCsr_mod_names = list(_labWeb['y'].keys())

        ## Administrator
        _s̆ᵂᵉᵇ = [ ## initial (true) state
            _labAdm['s̆']['s̆ᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'), 
            _labAdm['s̆']['s̆ᵂᵉᵇ₂'].index('DESCRIPTION_LINES')
        ]
        logger.info(f"Initial Administrator state: {_s̆ᵂᵉᵇ}")

        _yᵂᵉᵇ = [ ## initial observation
            _labAdm['y']['yᵂᵉᵇ₁'].index('UNKNOWN_OBS'), 
            _labAdm['y']['yᵂᵉᵇ₂'].index('CALL_TO_ACTION_OBS'),
            _labAdm['y']['yᵂᵉᵇ₃'].index('SITE_LINKS_OBS')
        ]
        logger.info(f"Initial Administrator observation: {_yᵂᵉᵇ}")

        ## Website
        _s̆ᶜˢʳ = [ ## initial (true) state
            _labWeb['s̆']['s̆ᶜˢʳ₁'].index('BUSINESS_HOURS'),
            _labWeb['s̆']['s̆ᶜˢʳ₂'].index('EAST')
        ]
        logger.info(f"Initial Website state: {_s̆ᶜˢʳ}")

        _yᶜˢʳ = [ ## initial observation
            _labWeb['y']['yᶜˢʳ₁'].index('BUSINESS_HOURS_OBS'),
            _labWeb['y']['yᶜˢʳ₂'].index('EAST_OBS'),
            _labWeb['y']['yᶜˢʳ₃'].index('ENGLISH_OBS')
        ]
        logger.info(f"Initial Website observation: {_yᶜˢʳ}")

        ## Create value name lists
        _aAdm_val_names = [_labAdm['a'][cfn] for cfn in _aAdm_fac_names]
        _sWeb_val_names = [_labAdm['s'][sfn] for sfn in _sWeb_fac_names]
        _s̆Web_val_names = [_labAdm['s̆'][sfn] for sfn in _s̆Web_fac_names]
        _yWeb_val_names = [_labAdm['y'][omn] for omn in _yWeb_mod_names]

        _aWeb_val_names = [_labWeb['a'][cfn] for cfn in _aWeb_fac_names]
        _sCsr_val_names = [_labWeb['s'][sfn] for sfn in _sCsr_fac_names]
        _s̆Csr_val_names = [_labWeb['s̆'][sfn] for sfn in _s̆Csr_fac_names]
        _yCsr_val_names = [_labWeb['y'][omn] for omn in _yCsr_mod_names]

        # Initialize tracking variables
        _aAdm_facs = {k: [] for k in _labAdm['a'].keys()}
        _sAdm_facs = {k: [] for k in _labAdm['s'].keys()}
        _s̆Web_facs = {k: [] for k in _labAdm['s̆'].keys()}
        _yWeb_mods = {k: [] for k in _labAdm['y'].keys()}
        _qAdmIpiIs = []
        _GAdmNegs = []
        
        _aWeb_facs = {k: [] for k in _labWeb['a'].keys()}
        _sWeb_facs = {k: [] for k in _labWeb['s'].keys()}
        _s̆Csr_facs = {k: [] for k in _labWeb['s̆'].keys()}
        _yCsr_mods = {k: [] for k in _labWeb['y'].keys()}
        _qWebIpiIs = []
        _GWebNegs = []

        # Store results for analysis
        results = {
            'Administrator': {
                'actions': _aAdm_facs,
                'states': _sAdm_facs,
                'true_states': _s̆Web_facs,
                'observations': _yWeb_mods,
                'policies': _qAdmIpiIs,
                'free_energy': _GAdmNegs
            },
            'Website': {
                'actions': _aWeb_facs,
                'states': _sWeb_facs,
                'true_states': _s̆Csr_facs,
                'observations': _yCsr_mods,
                'policies': _qWebIpiIs,
                'free_energy': _GWebNegs
            }
        }

        for t in range(_T):
            logger.info(f"\nTime {t}:")

            ### act
            logger.debug('___act___')
            actionAdm = _agtAdm.sample_action()
            actionWeb = _agtWeb.sample_action()

            ### future
            logger.debug('___future___')
            qIpiI_Adm, GNeg_Adm = _agtAdm.infer_policies()
            qIpiI_Web, GNeg_Web = _agtWeb.infer_policies()
            _qAdmIpiIs.append(qIpiI_Adm)
            _GAdmNegs.append(GNeg_Adm)
            _qWebIpiIs.append(qIpiI_Web)
            _GWebNegs.append(GNeg_Web)

            ### next
            logger.debug('___next___')
            # Administrator's actions influence Website's true states
            for sfi, sf in enumerate(_s̆ᵂᵉᵇ):
                _s̆ᵂᵉᵇ[sfi] = sample(_B̆ᴬᵈᵐ[sfi][:, sf, int(actionAdm[sfi])])
            for sfi, sfn in enumerate(_s̆Web_fac_names):
                _s̆Web_facs[sfn].append(_s̆Web_val_names[sfi][_s̆ᵂᵉᵇ[sfi]])

            # Website's actions influence Consumer's true states
            for sfi, sf in enumerate(_s̆ᶜˢʳ):
                _s̆ᶜˢʳ[sfi] = sample(_B̆ᵂᵉᵇ[sfi][:, sf, int(actionWeb[sfi])])
            for sfi, sfn in enumerate(_s̆Csr_fac_names):
                _s̆Csr_facs[sfn].append(_s̆Csr_val_names[sfi][_s̆ᶜˢʳ[sfi]])

            ### observe
            logger.debug('___observe___')
            # Generate observations from Website's true states for Administrator
            for omi, _ in enumerate(_yᵂᵉᵇ):
                if len(_s̆ᵂᵉᵇ) == 1:
                    _yᵂᵉᵇ[omi] = sample(_Ăᴬᵈᵐ[omi][:, _s̆ᵂᵉᵇ[0]])
                elif len(_s̆ᵂᵉᵇ) == 2:
                    _yᵂᵉᵇ[omi] = sample(_Ăᴬᵈᵐ[omi][:, _s̆ᵂᵉᵇ[0], _s̆ᵂᵉᵇ[1]])
                else:
                    logger.error(f'ERROR: {len(_s̆ᵂᵉᵇ)=} not handled!')
            for ymi, ymn in enumerate(_yWeb_mod_names):
                _yWeb_mods[ymn].append(_yWeb_val_names[ymi][_yᵂᵉᵇ[ymi]])

            # Generate observations from Consumer's true states for Website
            for omi, _ in enumerate(_yᶜˢʳ):
                if len(_s̆ᶜˢʳ) == 1:
                    _yᶜˢʳ[omi] = sample(_Ăᵂᵉᵇ[omi][:, _s̆ᶜˢʳ[0]])
                elif len(_s̆ᶜˢʳ) == 2:
                    _yᶜˢʳ[omi] = sample(_Ăᵂᵉᵇ[omi][:, _s̆ᶜˢʳ[0], _s̆ᶜˢʳ[1]])
                else:
                    logger.error(f'ERROR: {len(_s̆ᶜˢʳ)=} not handled!')
            for ymi, ymn in enumerate(_yCsr_mod_names):
                _yCsr_mods[ymn].append(_yCsr_val_names[ymi][_yᶜˢʳ[ymi]])

            ### infer
            logger.debug('___infer___')
            belief_state_Adm = _agtAdm.infer_states(_yᵂᵉᵇ)
            belief_state_Web = _agtWeb.infer_states(_yᶜˢʳ)
            for sfi, sfn in enumerate(_sWeb_fac_names):
                _sAdm_facs[sfn].append(_labAdm['s'][sfn][int(np.argmax(belief_state_Adm[sfi].round(3).T))])
            for sfi, sfn in enumerate(_sCsr_fac_names):
                _sWeb_facs[sfn].append(_labWeb['s'][sfn][int(np.argmax(belief_state_Web[sfi].round(3).T))])

        return results
    except Exception as e:
        logger.error(f"Error in run_simulation: {e}")
        logger.error(traceback.format_exc())
        raise

def save_results(results, config_summary, analysis_dir):
    """Save simulation results and configuration."""
    try:
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Save results
        results_file = os.path.join(analysis_dir, 'simulation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, default=convert_to_serializable, indent=2)
        logger.info(f"Saved simulation results to {results_file}")

        # Save configuration
        config_file = os.path.join(analysis_dir, 'configuration.json')
        with open(config_file, 'w') as f:
            json.dump(config_summary, f, indent=2)
        logger.info(f"Saved configuration to {config_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        logger.error(traceback.format_exc())
        raise

def save_visualizations(T, aAdm_facs, sAdm_facs, s̆Web_facs, yWeb_mods,
                       aWeb_facs, sWeb_facs, s̆Csr_facs, yCsr_mods,
                       GAdmNegs, GWebNegs, labAdm, labWeb, figures_dir):
    """Save visualization plots for the simulation results."""
    try:
        # Plot AWS interaction
        from pymdp.multiagent.LL_AWC_Visualization import plot_aws_interaction
        plot_aws_interaction(T=T,
                           aAdm_facs=aAdm_facs, sAdm_facs=sAdm_facs,
                           s̆Web_facs=s̆Web_facs, yWeb_mods=yWeb_mods,
                           aWeb_facs=aWeb_facs, sWeb_facs=sWeb_facs,
                           s̆Csr_facs=s̆Csr_facs, yCsr_mods=yCsr_mods,
                           labAdm=labAdm, labWeb=labWeb,
                           save_dir=figures_dir)
        logger.info("Successfully generated AWS interaction plot")

        # Plot free energy
        plt.figure(figsize=(10, 5))
        plt.plot(GAdmNegs, label='Administrator')
        plt.plot(GWebNegs, label='Website')
        plt.xlabel('Time Step')
        plt.ylabel('Negative Free Energy')
        plt.title('Free Energy Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'free_energy.png'))
        plt.close()
        logger.info("Successfully generated free energy plot")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        logger.error(traceback.format_exc())
        raise

def get_config_summary():
    """Get a summary of the current configuration."""
    try:
        config = {
            "simulation": {
                "timesteps": 25,  # _T value
                "agents": ["Administrator", "Website"]
            },
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        logger.info("Successfully generated configuration summary")
        return config
    except Exception as e:
        logger.error(f"Error generating configuration summary: {e}")
        logger.error(traceback.format_exc())
        raise