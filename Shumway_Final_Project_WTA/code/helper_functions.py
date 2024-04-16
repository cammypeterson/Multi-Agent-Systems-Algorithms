import numpy as np
import copy

def calc_EV(agents, targets):
    exp_val = 0
    for target in targets.values():
        kill_prob = target.calc_kill_prob([agent for agent in agents.values() if hasattr(agent, "target") and agent.target.id == target.id])
        exp_val += (1 - kill_prob)*target.value
        
    return exp_val

# Assumes same comms range for all agents
def update_adj_matrix(n, agents, comms_range):
    A = np.identity(n)
    for i in agents:
        for j in agents:
            if np.linalg.norm(agents[i].state[:2] - agents[j].state[:2]) <= comms_range:
                A[i, j] = 1
                A[j, i] = 1
    
    return A

def communicate(A, agents):
    for i in range(len(agents) - 1): # runs a communication round N - 1 times
        # dist_matrix = floyd_warshall(A)
        beliefs_dict = {}
        for agent_id in agents:
            beliefs_dict[agent_id] = copy.copy(agents[agent_id].belief)
            
        for agent1_id in agents:
            for agent2_id in agents:
                if agent1_id != agent2_id and A[agent1_id, agent2_id] != 0:
                    agents[agent2_id].receive_belief(beliefs_dict[agent1_id])
                    
    for agent in agents.values():
        agent.belief.reset_hops(agent.id) # this resets all hops (except agent's own) to infinity so that in the next round of communication, proper updates happen

# returns a list of agents ordered from highest to lowest weapon effectiveness
def order_agents(agents):
    agent_list = [agent for agent in agents.values()]
    
    # return list sorted in descending order by weapon effectiveness
    agent_list.sort(reverse=True, key=lambda agent:agent.weapon_effectiveness) 
    return agent_list

def target_assignment(A, agents, cost_function):
    agent_list = order_agents(agents)
    for i in range(len(agent_list)):
        agent_list[i].select_target('greedy', cost_function)
        communicate(A, agents)
        # TODO: call commmunicate here
