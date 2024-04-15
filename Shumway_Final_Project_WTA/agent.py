import numpy as np
import copy
from belief import Belief

class Agent:
    def __init__(self, id, pos, heading, agent_params, weapon_effectiveness_dict, target_dict, Ts):
        # self.max_glide_ratio = 1.
        # self.weapon_effectiveness = 0.75 # TODO: may be on an agent-target basis
        # self.attrition = 0.75 # TODO: may be on an agent-target basis
        # self.pa = 0.00004
        self.id = id
        self.state = np.array([pos.item(0), pos.item(1), agent_params["agent_velocity"], heading])
        
        self.alt = agent_params["agent_spawn_alt"]
        self.max_psidot = agent_params["max_psidot"]
        self.max_glide_ratio = agent_params["max_glide_ratio"]
        self.da = agent_params["num_attrition_sections"]
        self.pa = agent_params["pa"]
        self.collision_buffer = agent_params["collision_buffer"]
        
        self.Ts = Ts
        self.kp_psi = 3. # TODO: tune
        
        self.weapon_effectiveness = weapon_effectiveness_dict[id]
        
        self.prev_state = copy.copy(self.state)
        
        self.target_dict= target_dict
        
        self.belief = Belief(weapon_effectiveness_dict, target_dict.keys())
            
    def assign_target(self, target):
        if self.is_reachable(target):
            self.target = target
            self.attrition_prob = self.calc_attrition(self.target)
            self.belief.update_agent_estimate(self.id, self.target.id, self.attrition_prob, num_hops = 0)
            return True
        return False
    
    def update_estimates(self):
        self.belief.target_kill_prob = self.calc_all_kill_probabilities(self.belief.target_kill_prob.keys(), self.belief.agent_estimates)
        
    def is_reachable(self, target):
        v = self.state[2]
        zdot = v/np.linalg.norm(self.state[:2] - target.pos)*self.alt

        if abs(v/zdot) > self.max_glide_ratio:
            return False
        else:
            return True
    
    def decision(self, probability):
        rand = np.random.random()
        return rand < probability
    
    def receive_belief(self, rec_belief):
        for est_id, est_values in rec_belief.agent_estimates.items():
            if est_values['num_hops'] + 1 < self.belief.agent_estimates[est_id]['num_hops']:
                    self.belief.update_agent_estimate(est_id, est_values['assignment'], est_values['attrition_probability'], est_values['num_hops'] + 1)

    
    def check_collision(self):
        if np.linalg.norm(self.target.pos - self.state[:2]) < self.collision_buffer:
            return True, self.decision(self.weapon_effectiveness)
        
        # check to see if agents collided between time steps
        relative_speed = np.linalg.norm(np.zeros(2) - self.state[2]) # assumes target is stationary
        check_buffer = relative_speed*self.Ts # this is the minimum distance the simulation at this time step is able to detect
        
        if np.linalg.norm(self.target.pos - self.state[:2]) < check_buffer*1.1: # if the distance between the two agents is less than the buffer (plus 50% cushion)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot([self.prev_state[0], self.state[0]], [self.prev_state[1], self.state[1]])
            # plt.scatter(self.target.pos[0], self.target.pos[1])
            # plt.show(block=False)
            
            new_ts = int(relative_speed/self.collision_buffer) + 1
            agent1_pos_list = np.linspace(self.prev_state[:2], self.state[:2], new_ts)
            agent2_pos_list = np.array([self.target.pos for i in range(len(agent1_pos_list))]) # assumes stationary target
            
            distance = np.zeros(len(agent1_pos_list))
            for i in range(new_ts):
                distance[i] = np.linalg.norm(agent1_pos_list[i] - agent2_pos_list[i]) # only needed for plotting
                if np.linalg.norm(agent1_pos_list[i] - agent2_pos_list[i]) < self.collision_buffer:
                    return True, self.decision(self.weapon_effectiveness)
        
        return False, False
    
    def calc_attrition(self, target):
        dij = np.linalg.norm(target.pos - self.state[:2]) # this is 2D distance
        d_int = dij/self.da 
        return  1 - (1 - self.pa)**d_int
        
        # update current estimate of self 
        self.belief.update_agent_estimate(self.id, target.id, self.attrition_prob, num_hops = 0)
    
    def check_attrition(self):
        self.attrition_prob = self.calc_attrition(self.target)
        return self.decision(self.attrition_prob)
            
        
    def update_dynamics(self):
        target_pos = self.target.pos[:2]
        seeker_pos = self.state[:2]
        
        crs_cmd = np.arctan2(target_pos.item(1) - seeker_pos.item(1), target_pos.item(0) - seeker_pos.item(0))
        
        self.RK4([crs_cmd])
        
        self.state[3] = self.bound_angle(self.state[3])
        
    def bound_angle(self, angle):
        while angle > np.pi:
            angle -= 2*np.pi
        while angle <= -np.pi:
            angle += 2*np.pi
        return angle
    
    def derivatives(self, state, crs_cmd):
        
        n, e, V, psi = state
        
        n_dot = V*np.cos(psi)
        e_dot = V*np.sin(psi)
        V_dot = 0
        
        crs_error = self.bound_angle(crs_cmd - psi)
        psi_dot = self.kp_psi*crs_error
            
        psi_dot = self.saturate(psi_dot)
        
        return np.array([n_dot, e_dot, V_dot, psi_dot])
        
    def saturate(self, psidot_cmd):
        if abs(psidot_cmd) > self.max_psidot:
            return np.sign(psidot_cmd)*self.max_psidot
        else:
            return psidot_cmd
        
    def RK4(self, inputs):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = self.derivatives(self.state, *inputs)
        k2 = self.derivatives(self.state + self.Ts/2.*k1, *inputs)
        k3 = self.derivatives(self.state + self.Ts/2.*k2, *inputs)
        k4 = self.derivatives(self.state + self.Ts*k3, *inputs)
        self.state += self.Ts/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def calc_kill_prob(self, seekers):
        product = 1
        
        for agent in seekers: # seekers is a list of seekers targeting the current target
            product *= (1 - agent["weapon_effectiveness"] + agent["weapon_effectiveness"]*agent["attrition_probability"])
            
        return 1 - product
            
    def calc_all_kill_probabilities(self, target_list, agent_estimates):
        target_kill_prob = {}
        for target_id in target_list:
            seekers = [agent_estimates[agent_est] for agent_est in agent_estimates if agent_estimates[agent_est]['assignment'] == target_id] # get a list of agents who you belief are targeting the target
            target_kill_prob[target_id] = self.calc_kill_prob(seekers)
        
        return target_kill_prob
    
    def calc_cost(self, agent_states, cost_function):
        cost = 0
        target_kill_probabilities = {}
        
        for target_id in self.target_dict:
            target_kill_probabilities[target_id] = 0
        target_kill_probabilities = self.calc_all_kill_probabilities(self.target_dict.keys(), agent_states)
        
        if cost_function == "traditional":
            for target_id in target_kill_probabilities:
                cost += (1 - target_kill_probabilities[target_id])*self.target_dict[target_id].value
        elif cost_function == "sufficiency threshold":
            alpha = 1
            for target_id in target_kill_probabilities:
                if target_kill_probabilities[target_id] > self.target_dict[target_id].des_kill_prob:
                    cost += 0
                else:
                    cost += (self.target_dict[target_id].des_kill_prob - target_kill_probabilities[target_id])/(1 -  self.target_dict[target_id].des_kill_prob)**alpha
        elif cost_function == "tiered":
            tiers = {}
            for target_id in target_kill_probabilities:
                des_kill_prob = self.target_dict[target_id].des_kill_prob
                if des_kill_prob in tiers:
                    tiers[des_kill_prob].append(target_id)
                else:
                    tiers[des_kill_prob] = [target_id]
            
            for tier_val in tiers.keys(): # for each tier
                highest_tier = True
                for other_tier_val in tiers.keys(): 
                    if tier_val != other_tier_val:  
                        if other_tier_val > tier_val: # for all higher tiers
                            highest_tier = False
                            penalty_flag = False
                            for higher_tier_id in tiers[other_tier_val]:
                                if target_kill_probabilities[higher_tier_id] < self.target_dict[higher_tier_id].des_kill_prob:
                                    penalty_flag = True
                                    break
                                
                            tier_dict = {}
                            for id in agent_states:
                                if id in tiers[tier_val]:
                                    tier_dict[id] = agent_states[id]    
                                
                            if penalty_flag:
                                #calculate penalty 
                                # max_cost = 0
                                # pot_agent_states = copy.copy(tier_dict)
                                # if not self.id in pot_agent_states:
                                #     pot_agent_states[self.id] = agent_states[self.id]
                                # for target_id in tier_dict:
                                #     if self.is_reachable(self.target_dict[target_id]):
                                #         pot_agent_states[self.id]['assignment'] = target_id
                                #         pot_agent_states[self.id]['attrition_probability'] = self.calc_attrition(self.target_dict[target_id])
                                #         cost_temp = self.calc_cost(pot_agent_states, 'sufficiency threshold') 
                                #         if cost_temp > max_cost:
                                #             max_cost = cost_temp
                                
                                penalty = 1000 #max_cost*1.5
                                cost += penalty
                            else:
                                cost += self.calc_cost(tier_dict, "sufficiency threshold")
                if highest_tier:
                    tier_dict = {}
                    for id in agent_states:
                        if agent_states[id]['assignment'] in tiers[tier_val]:
                            tier_dict[id] = agent_states[id]   
                    
                    cost += self.calc_cost(tier_dict, "sufficiency threshold")       
        elif cost_function == "completion":
            target_id = agent_states[self.id]['assignment']
            des_kill_prob = self.target_dict[target_id].des_kill_prob
            agent_states_copy = copy.deepcopy(agent_states)
            agent_states_copy[self.id]['assignment'] = None
            if self.belief.agent_estimates[self.id]['assignment'] is None: 
                target_kill_prob_wo_self = self.calc_all_kill_probabilities(self.target_dict.keys(), agent_states_copy)
                if target_kill_prob_wo_self[target_id] >= des_kill_prob:
                    if self.belief.agent_estimates[self.id]['assignment'] == target_id:
                        return 0
                    else:
                        return 1000
            elif target_kill_probabilities[target_id] >= des_kill_prob:
                if target_kill_probabilities[target_id] >= des_kill_prob:
                    if self.belief.agent_estimates[self.id]['assignment'] == target_id:
                        return 0
                    else:
                        return 1000
            return (np.log(1 - des_kill_prob) - np.log(1 - target_kill_probabilities[target_id]))/np.log(1 - self.weapon_effectiveness + self.weapon_effectiveness*agent_states[self.id]['attrition_probability'])
        else: 
            raise ValueError("Invalid cost function type specified")
            
        
        return cost
    
    def select_target(self, method = 'greedy', cost_function = 'completion'):
        if method == 'greedy':
            self.select_target_greedy(cost_function)
        else:
            raise ValueError("Invalid select target method")
        
    def select_target_greedy(self, cost_function):
        min_cost = np.inf
        min_cost_assignment = ''
        pot_agent_estimates = copy.deepcopy(self.belief.agent_estimates)
        for target_id in self.belief.target_kill_prob.keys():
            if self.is_reachable(self.target_dict[target_id]):
                pot_agent_estimates[self.id]['assignment'] = target_id
                pot_agent_estimates[self.id]['attrition_probability'] = self.calc_attrition(self.target_dict[target_id])
                cost = self.calc_cost(pot_agent_estimates, cost_function) 
                if cost < min_cost:
                    min_cost = cost
                    min_cost_assignment = target_id
                
        self.assign_target(self.target_dict[min_cost_assignment])