import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

def plot_aws_interaction(T, aAdm_facs, sAdm_facs, s̆Web_facs, yWeb_mods, 
                        aWeb_facs, sWeb_facs, s̆Csr_facs, yCsr_mods,
                        labAdm, labWeb):
    """Plot Administrator-Website interaction visualization"""
    
    colors = [
        {'NULL_ACT':'black'}, # aAdm_1
        {'BROAD_MATCH_ACT':'red', 'PHRASE_MATCH_ACT':'green', 'EXACT_MATCH_ACT': 'blue'}, # aAdm_2
        {'EXPANDED_TEXT_ADS':'orange', 'RESPONSIVE_SEARCH_ADS':'purple'}, # sWeb_1
        {'DESCRIPTION_LINES':'red', 'DISPLAY_URL':'green', 'CALL_TO_ACTION': 'blue'}, # sWeb_2
        {'EXPANDED_TEXT_ADS':'orange', 'RESPONSIVE_SEARCH_ADS':'purple'}, # qIsIWeb_1
        {'DESCRIPTION_LINES':'red', 'DISPLAY_URL':'green', 'CALL_TO_ACTION': 'blue'}, # qIsIWeb_2
        {'EXPANDED_TEXT_ADS_OBS':'orange', 'RESPONSIVE_SEARCH_ADS_OBS':'purple', 'UNKNOWN_OBS':'pink'}, # yWeb_1
        {'DESCRIPTION_LINES_OBS':'red', 'DISPLAY_URL_OBS':'green', 'CALL_TO_ACTION_OBS': 'blue'}, # yWeb_2
        {'SITE_LINKS_OBS':'red', 'CALLOUTS_OBS':'green', 'STRUCTURED_SNIPPETS_OBS': 'blue'} # yWeb_3
    ]

    ylabel_size = 12
    msi = 7  # markersize for Line2D
    siz = (msi/2)**2 * np.pi  # size for scatter

    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(9, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    ax = [fig.add_subplot(gs[i]) for i in range(9)]

    # Plot Administrator actions
    plot_actions(ax[0], T, aAdm_facs['aᴬᵈᵐ₁'], colors[0], msi, siz, 'a^{Adm}_{1t}, NULL')
    plot_actions(ax[1], T, aAdm_facs['aᴬᵈᵐ₂'], colors[1], msi, siz, 'a^{Adm}_{2t}, SET\_MATCH\_TYPE\_ACTION')

    # Plot Website states and beliefs
    plot_states(ax[2], T, sAdm_facs['sᵂᵉᵇ₁'], colors[2], msi, siz, 's^{Web}_{1t}, AD\_TYPE')
    plot_states(ax[3], T, sAdm_facs['sᵂᵉᵇ₂'], colors[3], msi, siz, 's^{Web}_{2t}, AD\_COPY\_CREATION')

    # Plot inferred states
    plot_states(ax[4], T, sAdm_facs['sᵂᵉᵇ₁'], colors[4], msi, siz, 'q(s)^{Web}_{1t}, AD\_TYPE')
    plot_states(ax[5], T, sAdm_facs['sᵂᵉᵇ₂'], colors[5], msi, siz, 'q(s)^{Web}_{2t}, AD\_COPY\_CREATION')

    # Plot observations
    plot_observations(ax[6], T, yWeb_mods['yᵂᵉᵇ₁'], colors[6], msi, siz, 'y^{Web}_{1t}, AD\_TYPE\_OBS')
    plot_observations(ax[7], T, yWeb_mods['yᵂᵉᵇ₂'], colors[7], msi, siz, 'y^{Web}_{2t}, AD\_COPY\_CREATION\_OBS')
    plot_observations(ax[8], T, yWeb_mods['yᵂᵉᵇ₃'], colors[8], msi, siz, 'y^{Web}_{3t}, AD\_EXTENSIONS\_OBS')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()

def plot_actions(ax, T, actions, color_map, msi, siz, ylabel):
    """Plot action trajectories"""
    y_pos = 0
    for t, s in enumerate(actions):
        ax.scatter(t, y_pos, color=color_map[s], s=siz)
    ax.set_ylabel(f'${ylabel}$', rotation=0, fontweight='bold', fontsize=12)
    ax.set_yticks([])
    ax.set_xticklabels([])
    leg_items = [Line2D([0],[0], marker='o', color='w', 
                       markerfacecolor=color, markersize=msi, label=label)
                for label, color in color_map.items()]
    ax.legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), 
             loc='center left', borderaxespad=0, labelspacing=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_states(ax, T, states, color_map, msi, siz, ylabel):
    """Plot state trajectories"""
    y_pos = 0
    for t, s in enumerate(states):
        ax.scatter(t, y_pos, color=color_map[s], s=siz)
    ax.set_ylabel(f'${ylabel}$', rotation=0, fontweight='bold', fontsize=12)
    ax.set_yticks([])
    ax.set_xticklabels([])
    leg_items = [Line2D([0],[0], marker='o', color='w', 
                       markerfacecolor=color, markersize=msi, label=label)
                for label, color in color_map.items()]
    ax.legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), 
             loc='center left', borderaxespad=0, labelspacing=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_observations(ax, T, observations, color_map, msi, siz, ylabel):
    """Plot observation trajectories"""
    y_pos = 0
    for t, s in enumerate(observations):
        ax.scatter(t, y_pos, color=color_map[s], s=siz)
    ax.set_ylabel(f'${ylabel}$', rotation=0, fontweight='bold', fontsize=12)
    ax.set_yticks([])
    if ylabel.endswith('3t}'):  # Last subplot
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('$\\mathrm{time,}\\ t$', fontweight='bold', fontsize=12)
    else:
        ax.set_xticklabels([])
    leg_items = [Line2D([0],[0], marker='o', color='w', 
                       markerfacecolor=color, markersize=msi, label=label)
                for label, color in color_map.items()]
    ax.legend(handles=leg_items, bbox_to_anchor=(1.05, 0.5), 
             loc='center left', borderaxespad=0, labelspacing=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)