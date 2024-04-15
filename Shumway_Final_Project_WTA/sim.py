# TODO: check/debug (not bugs found yet) greedy function and traditional cost function
# TODO: tune attrition (da)?

from agent import Agent
from target import Target
import numpy as np
import matplotlib.pyplot as plt
import plotter

from helper_functions import update_adj_matrix, target_assignment

################# SIMULATION PARAMETERS ########################

cost_function = 'traditional' # which cost function to use ("traditional", "sufficiency threshold", "tiered", or "completion")
animate = False # if true, plots an animation of agent/target states
save_frames = False # if true, animation frames are saved to the 'frames/' directory rather than displayed
                    # If 'animate' is false, this boolean does not matter
np.random.seed(9) # set the seed to repeat runs. Comment out to get different results each time

target_spawn_dim = 750. # length (m) of the square within which the targets spawn randomly
agent_spawn_dim = 750. # length (m) of the square within which the agents spawn randomly
agent_target_spawn_dist = 2500. # distance (m) between agent spawn square and target spawn square

weapon_effectiveness = 0.7 # effectiveness of weapons (assumed to be the same for all)

comms_range = 600. # communication range of agents (m)

agent_params = {"agent_spawn_alt": 500., # altitude of agents (m)
                "agent_velocity": 50.,
                "max_psidot": 1.265,
                "max_glide_ratio": 6.,
                "num_attrition_sections": 100,
                "pa": 0.00004,
                "collision_buffer": 5.0 # distance (m) within which a collision is defined
                } 

round_ts = 1e-1
end_time = 100.0

num_targets = 6

num_agents = 8

################################################################

# init agents
weapon_effectiveness_dict = {}
for i in range(num_agents):
    weapon_effectiveness_dict[i] = 0.7 # For now, this is for all agents. Some simulations may need to create is separately

target_colors = ['white', 'navy', 'royalblue', 'seagreen', 'limegreen', 'red', 'lightcoral', 'blue', 'green', 'cyan', 'yellow', 'orange', 'purple']

# init targets   
active_targets = {}
for i in range(num_targets):
    pos = np.array([np.random.uniform(high = target_spawn_dim), np.random.uniform(high = target_spawn_dim)])
    if i < 2:
        active_targets[i] = Target(i, pos, 0.9, value = 0.9) # hardcoded desired kill probability
    elif i < 4:
        active_targets[i] = Target(i, pos, 0.8, value = 0.8) # hardcoded desired kill probability
    else:
        active_targets[i] = Target(i, pos, 0.7, value = 0.7) # hardcoded desired kill probability


# init agents and assign targets 
active_agents = {}
for i in range(num_agents):
    pos = np.array([np.random.uniform(high = agent_spawn_dim), np.random.uniform(high = agent_spawn_dim) + agent_target_spawn_dist])
    
    heading = np.random.uniform(low = -np.pi, high = np.pi)
    
    agent = Agent(i, pos, heading, agent_params, weapon_effectiveness_dict, active_targets, round_ts) # assumes all agents have same effectiveness, but a different value can be put here if needed
    
    active_agents[i] = agent

# init communications model
A = update_adj_matrix(num_agents, active_agents, comms_range)
target_assignment(A, active_agents, cost_function)

inactive_targets = {} 
inactive_agents = {}

sim_time = 0

# init charts
sim_time_hist = [sim_time]
target_kill_probabilities_hist = {}
agent_assignment_hist = {}
for target_id in active_targets:
    target_kill_probabilities_hist[target_id] = [0]
for agent_id in active_agents:
    agent_assignment_hist[agent_id] = [active_agents[agent_id].target.id]

if animate:
    anim_plt_init = False
    anim_fig, anim_ax = plt.subplots()
    
    
# simulation loopo
while sim_time < end_time and len(active_targets) != 0:
    A = update_adj_matrix(num_agents, active_agents, comms_range)
    target_assignment(A, active_agents, cost_function)
    sim_time += round_ts
    sim_time_hist.append(sim_time)
    
    if animate:
        anim_ax.cla()
        
    for id, agent in active_agents.items():
        agent.update_dynamics() # update each agent's dynamics
        
        # check for agent/target collisions
        agent_collided, target_destroyed = agent.check_collision() 
        if agent_collided:
            inactive_agents[id] = agent
            print(f"[{sim_time:.2f}]: Agent {id} has collided with Target {agent.target.id}")
            if target_destroyed:
                inactive_targets[agent.target.id] = agent.target
                print(f"[{sim_time:.2f}]: Target {agent.target.id} has been destroyed")
        
        # check if agents are attrited
        elif agent.check_attrition():
            inactive_agents[id] = agent
            print(f"[{sim_time:.2f}]: Agent {id} has been attrited")
            
        agent.update_estimates() # update estimates of each agent based on communication protocol
        
        # plot animation frame
        if animate:
            agent_pos = agent.state[:2]
            agent_heading = agent.state.item(3)
            
            # plot agent position
            anim_ax.scatter(agent_pos.item(1), agent_pos.item(0), color = 'orange')
            
            # plot agent heading direction
            line_len = 50.
            anim_ax.plot([agent_pos.item(1), agent_pos.item(1) + line_len*np.sin(agent_heading)], [agent_pos.item(0), agent_pos.item(0) + line_len*np.cos(agent_heading)], color = 'green')
            
            # plot connection between agent and target
            anim_ax.plot([agent_pos.item(1), agent.target.pos.item(1)], [agent_pos.item(0), agent.target.pos.item(0)], color = 'c')
    
    # remove any inactive agents from the active list
    for id in inactive_agents.keys():
        if id in active_agents:
            
            # Make all agents aware of destroyed target
            for agent in active_agents.values():
                del agent.belief.agent_estimates[id]

            del active_agents[id]
               
    
    # remove any inactive agents from the active list
    for id in inactive_targets.keys():
        if id in active_targets:
            active_targets[id].des_kill_prob = 0 # set target's desired kill prob to zero (maybe not necessary since there is an "inactive_targets" dict)
            
            # Make all agents aware of destroyed target
            for agent in active_agents.values():
                del agent.belief.target_kill_prob[id]
                
            del active_targets[id]
    
    
    # save data for plotting at the end
    if len(active_agents) > 0:
        # save assignment history
        for agent_id in agent_assignment_hist:
            if agent_id in active_agents:
                agent_assignment_hist[agent_id].append(active_agents[agent_id].target.id)
        
        # save target kill probability history
        target_kill_probabilities = active_agents[list(active_agents.keys())[0]].belief.target_kill_prob
        for target_id in target_kill_probabilities_hist:
            if target_id in target_kill_probabilities:
                target_kill_probabilities_hist[target_id].append(target_kill_probabilities[target_id])
        
        # plot target positions on animation frame
        if animate:
            for target in active_targets.values():
                if target.id < 2:
                    anim_ax.scatter(target.pos.item(1), target.pos.item(0), color = target_colors[target.id + 1])
                elif target.id < 4:
                    anim_ax.scatter(target.pos.item(1), target.pos.item(0), color = target_colors[target.id + 1])
                else:
                    anim_ax.scatter(target.pos.item(1), target.pos.item(0), color = target_colors[target.id + 1])
            
            anim_ax.set_aspect("equal")
            if not anim_plt_init:
                xlim = anim_ax.get_xlim()
                ylim = anim_ax.get_ylim()
                anim_plt_init = True
            else:
                anim_ax.set_xlim(xlim)
                anim_ax.set_ylim(ylim)
            
            anim_ax.text(xlim[1]*0.87, ylim[1]*1.02, f"t = {sim_time:.2f}s")    
        
        
            if save_frames: # save frame instead of displaying it
                if sim_time < 1:
                    file_string = f'/frame_00{int(sim_time*10)}.png'
                elif sim_time < 10:
                    file_string = f'/frame_0{int(sim_time*10)}.png'
                elif sim_time < 100:
                    file_string = f'/frame_{int(sim_time*10)}.png'
                plt.savefig('frames/' + cost_function + file_string) 
            else: # display frame
                plt.pause(round_ts)
    else:
        break

plotter.plot_achieved_pk(target_kill_probabilities_hist, num_targets) # plot history of target kill probabilities

plotter.plot_agent_assignments(agent_assignment_hist, num_agents, num_targets) # plot assignment history