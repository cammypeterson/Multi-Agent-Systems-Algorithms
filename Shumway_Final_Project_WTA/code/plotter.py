import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def plot_achieved_pk(target_kill_probabilities_hist, num_targets):
    max_len = 0
    for target_id, kill_prob_hist in target_kill_probabilities_hist.items(): 
        temp_len = len(kill_prob_hist)
        if temp_len > max_len:
            max_len = temp_len


    
    p_values = np.zeros((len(target_kill_probabilities_hist), int(np.round(max_len/10))))

    for target_id, kill_prob_hist in target_kill_probabilities_hist.items(): 
        temp_len = int(np.round(len(kill_prob_hist)/10))
        p_values[target_id, :temp_len] = [kill_prob_hist[10*i + 1] for i in range(temp_len)]
        
        
    # Plotting
    plt.figure(figsize=(p_values.shape[1]/4, num_targets))

    colors = [(0, 'white'), (0.4, 'white'), (0.7, 'blue'), (0.9, 'purple'), (1, 'red')]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot the heatmap
    # plt.imshow(p_values, cmap=custom_cmap, aspect='auto', interpolation='nearest', extent=[0, , 0, num_targets])
    plt.imshow(p_values, cmap=custom_cmap, aspect='auto', interpolation='nearest', extent=[0, p_values.shape[1], 0, num_targets], vmin=0, vmax=1)


    # Customize plot
    plt.colorbar(label='p value')
    # norm = BoundaryNorm([0.5, 0.75, 1.0], len(colors))
    # cbar.set_ticks([0.5, 0.75, 1.0])
    # cbar.set_label('p value')
    plt.xlabel('Time (s))')
    # plt.ylabel('Agent')
    plt.title('Greedy Search - Achieved Pk')

    # Set ticks and labels
    x_ticks = np.arange(0, p_values.shape[1], 1)  # Show ticks every 5th time step
    plt.xticks(x_ticks, labels=['' if i % 5 != 0 else str(i) for i in range(p_values.shape[1])])

    # Set minor ticks for x-axis to align with grid lines
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))

    # Set ticks and labels for y-axis
    y_labels = [f'Target {num_targets - i}' for i in range(1, num_targets + 1)]
    y_ticks = np.arange(num_targets)
    plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)  # Hide original y-axis labels


    # Manually adjust y-axis labels to shift them up
    label_padding = 0.5  # Padding for shifting labels up
    for tick, label in zip(y_ticks, y_labels):
        plt.text(-0.01, tick + label_padding, label, ha='right', va='center', transform=plt.gca().get_yaxis_transform())

    # Add grid lines
    plt.grid(True, linestyle='-', linewidth=0.5, color='black')


    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    plt.savefig('achieved_pk_trad.png')

    # Optionally, you can also save in PDF format
    # plt.savefig('agent_values_heatmap.pdf')

    # Optionally, you can save with higher resolution (dpi)
    # plt.savefig('agent_values_heatmap.png', dpi=300)

    # Close the plot to release memory (optional)
    plt.close()
    
def plot_agent_assignments(agent_assignment_hist, num_agents, num_targets):
    
    max_len = 0
    for agent_id in agent_assignment_hist:
        temp_len = len(agent_assignment_hist[agent_id])
        if temp_len > max_len:
            max_len = temp_len
    
    assign_array = np.full((num_agents, int(np.round(max_len/10))), -1)
    
    for agent_id, assign_hist in agent_assignment_hist.items():
        temp_len = int(np.round(len(assign_hist)/10))
        assign_array[agent_id, :temp_len] = [assign_hist[10*i] for i in range(temp_len)]
       
    
    
     # Plotting
    plt.figure(figsize=(assign_array.shape[1]/4, num_agents))

    colors = ['white', 'navy', 'royalblue', 'seagreen', 'limegreen', 'red', 'lightcoral', 'blue', 'green', 'cyan', 'yellow', 'orange', 'purple']
    colors = colors[:num_targets + 1]
    cmap = ListedColormap(colors)

    # Plot the heatmap
    # plt.imshow(p_values, cmap=custom_cmap, aspect='auto', interpolation='nearest', extent=[0, , 0, num_targets])
    plt.imshow(assign_array, cmap=cmap, aspect='auto', interpolation='nearest', extent=[0, assign_array.shape[1], 0, num_agents], vmin=-1, vmax=num_targets)


    # Customize plot
    # plt.colorbar(label='Assigned Target')
    cbar = plt.colorbar(orientation='vertical', extend='neither')
    cbar.ax.invert_yaxis()
    
    # Set custom tick labels and positions
    ticks = np.linspace(-1, num_targets - 1, num_targets + 1)
    tick_labels = ["Target " + str(int(i)) for i in np.linspace(-1, num_targets - 1, num_targets + 1)]
    tick_labels[0] = "No target"

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    # Manually adjust tick label positions
    label_padding = 0.5  # Padding for shifting labels
    for tick, label in zip(ticks, tick_labels):
        cbar.ax.text(1.5, tick + label_padding, label, ha='left', va='center')
        
    # Remove the original tick labels
    cbar.ax.set_yticklabels([])
    
    # norm = BoundaryNorm([0.5, 0.75, 1.0], len(colors))
    # cbar.set_ticks([0.5, 0.75, 1.0])
    # cbar.set_label('p value')
    plt.xlabel('Time (s))')
    # plt.ylabel('Agent')
    plt.title('Current Target Assignment')

    # Set ticks and labels
    x_ticks = np.arange(0, assign_array.shape[1], 1)  # Show ticks every 5th time step
    plt.xticks(x_ticks, labels=['' if i % 5 != 0 else str(i) for i in range(assign_array.shape[1])])

    # Set minor ticks for x-axis to align with grid lines
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))

    # Set ticks and labels for y-axis
    y_labels = [f'Agent {num_agents - i}' for i in range(1, num_agents + 1)]
    y_ticks = np.arange(num_agents)
    plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)  # Hide original y-axis labels


    # Manually adjust y-axis labels to shift them up
    label_padding = 0.5  # Padding for shifting labels up
    for tick, label in zip(y_ticks, y_labels):
        plt.text(-0.01, tick + label_padding, label, ha='right', va='center', transform=plt.gca().get_yaxis_transform())


    # Add grid lines
    plt.grid(True, linestyle='-', linewidth=0.5, color='black')


    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    plt.savefig('current_target_trad.png')

    # Optionally, you can also save in PDF format
    # plt.savefig('agent_values_heatmap.pdf')

    # Optionally, you can save with higher resolution (dpi)
    # plt.savefig('agent_values_heatmap.png', dpi=300)

    # Close the plot to release memory (optional)
    plt.close()
