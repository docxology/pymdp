"""
Multiagent package for Active Web Campaign simulation.
"""

from pathlib import Path

# Package metadata
__version__ = '0.1.0'
__author__ = 'PyMDP Contributors'

# Package paths
PACKAGE_ROOT = Path(__file__).parent.absolute()

# Import core components
from .LL_Methods import (
    setup_paths,
    check_and_install_dependencies,
    check_pymdp_installation,
    import_dependencies,
    setup_output_directories,
    get_model_dimensions_from_labels,
    run_simulation,
    save_results,
    save_visualizations,
    get_config_summary
)

__all__ = [
    'setup_paths',
    'check_and_install_dependencies',
    'check_pymdp_installation',
    'import_dependencies',
    'setup_output_directories',
    'get_model_dimensions_from_labels',
    'run_simulation',
    'save_results',
    'save_visualizations',
    'get_config_summary',
    'PACKAGE_ROOT'
] 