import numpy as np


class CBBA_Network:
    def __init__(self, Nt, Nu, Lt, agents, tasks, graph):
        self.Nt = Nt
        self.Nu = Nu
        self.Lt = Lt
        self.agents = agents  # (Nu) len list of Agents
        self.tasks = tasks  # (Nt x dim) ndarray
        self.graph = graph  # (Nu x Nu) undirected connectivity graph

        self.step_num = 0
        return

    def step(self):
        self.step_num += 1
        for i in range(self.Nu):
            self.agents[i].build_bundle()

        for i in range(self.Nu):
            for j in range(self.Nu - i - 1):
                k = i + j + 1
                if self.graph[i, k] == 1:
                    yk = self.agents[k].y
                    zk = self.agents[k].z
                    sk = self.agents[k].s
                    yi = self.agents[i].y
                    zi = self.agents[i].z
                    si = self.agents[i].s
                    self.agents[i].receive_communication(k, yk, zk, sk, self.step_num)
                    self.agents[k].receive_communication(i, yi, zi, si, self.step_num)
        return

class Agent:
    def __init__(self, Nt, Nu, Lt, tasks, agent_num, pos_init, vel):
        self.Nt = Nt  # num tasks
        self.Nu = Nu  # num agents
        self.Lt = Lt  # max num of tasks per agent
        self.tasks = tasks  # tasks - (Nt x dim) ndarray
        self.agent_num = agent_num  # agent i
        self.pos = pos_init  # position of agent - (dim) ndarray
        self.vel = vel  # constant velocity of agent

        self.b = []  # task bundle
        self.p = []  # task path
        self.y = np.zeros((Nt))  # winning bids
        self.z = -1 * np.ones((Nt), dtype=int)  # winning agents
        self.s = np.zeros((Nu))  # most recent communication times

        self.path_score = 0
        return

    def build_bundle(self):
        while len(self.b) < self.Lt:
            c, n_opts = self.calc_task_scores()  # set used tasks to -1
            h = c > self.y
            if np.sum(h) == 0:
                break
            j_opt = np.argmax(np.multiply(c, h))
            n_opt = n_opts[j_opt]
            self.b.append(j_opt)
            self.p.insert(n_opt, j_opt)
            self.y[j_opt] = c[j_opt]
            self.z[j_opt] = self.agent_num
            self.path_score = self.score_path(self.p)
        return

    def receive_communication(self, k, yk, zk, sk, step_num):
        update_reset_tasks = []
        for j in range(self.Nt):
            if zk[j] == k:
                if self.z[j] == self.agent_num:
                    if yk[j] > self.y[j]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                elif self.z[j] == k:
                    self.update(j, yk[j], zk[j], update_reset_tasks)
                elif self.z[j] != -1:
                    m = self.z[j]
                    if sk[m] > self.s[m] or yk[j] > self.y[j]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                else:
                    self.update(j, yk[j], zk[j], update_reset_tasks)
            elif zk[j] == self.agent_num:
                if self.z[j] == self.agent_num:
                    continue
                elif self.z[j] == k:
                    self.reset(j, update_reset_tasks)
                elif self.z[j] != -1:
                    m = self.z[j]
                    if sk[m] > self.s[m]:
                        self.reset(j, update_reset_tasks)
                else:
                    continue
            elif zk[j] != -1:
                m = zk[j]
                if self.z[j] == self.agent_num:
                    if sk[m] > self.s[m] and yk[j] > self.y[j]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                elif self.z[j] == k:
                    if sk[m] > self.s[m]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                    else:
                        self.reset(j, update_reset_tasks)
                elif self.z[j] == m:
                    if sk[m] > self.s[m]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                elif self.z[j] != -1:
                    n = self.z[j]
                    if sk[m] > self.s[m] and sk[n] > self.s[n]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                    elif sk[m] > self.s[m] and yk[j] > self.y[j]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                    elif sk[n] > self.s[n] and self.s[m] > sk[m]:
                        self.reset(j, update_reset_tasks)
                else:
                    if sk[m] > self.s[m]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
            else:
                if self.z[j] == self.agent_num:
                    continue
                elif self.z[j] == k:
                    self.update(j, yk[j], zk[j], update_reset_tasks)
                elif self.z[j] != -1:
                    m = self.z[j]
                    if sk[m] > self.s[m]:
                        self.update(j, yk[j], zk[j], update_reset_tasks)
                else:
                    continue
        self.s = np.maximum(sk, self.s)
        self.s[k] = step_num
        self.s[self.agent_num] = 0
        self.update_bundle(update_reset_tasks)
        return

    def update_bundle(self, tasks):
        for task in tasks:
            if task in self.b:
                n_bar = 0
                for j in range(len(self.b)):
                    if self.z[self.b[j]] == -1 or self.z[self.b[j]] != self.agent_num:
                        n_bar = j
                        break
                bundle_len = len(self.b)
                for j in range(bundle_len - n_bar):
                    old_task = self.b[-1]
                    self.p.remove(old_task)
                    self.b.remove(old_task)
                    if j + n_bar < bundle_len - 1:
                        self.y[old_task] = 0
                        self.z[old_task] = -1
        self.path_score = self.score_path(self.p)
        return

    def update(self, j, yj, zj, update_reset_tasks):
        self.y[j] = yj
        self.z[j] = zj
        update_reset_tasks.append(j)

    def reset(self, j, update_reset_tasks):
        self.y[j] = 0
        self.z[j] = -1
        update_reset_tasks.append(j)

    def calc_task_scores(self):
        c = np.zeros((self.Nt))
        n_opts = []
        for j in range(self.Nt):
            if j not in self.b:
                c_opt, n_opt = self.find_task_place(j)
                c[j] = c_opt
                n_opts.append(n_opt)
            else:
                c[j] = -1
                n_opts.append(-1)
        return c, n_opts

    def find_task_place(self, j):
        c_best = 0
        n_best = -1
        for i in range(len(self.p) + 1):
            new_path = self.p.copy()
            new_path.insert(i, j)
            c = self.score_path(new_path) - self.path_score
            if c > c_best:
                c_best = c
                n_best = i
        return c_best, n_best

    def score_path(self, path):  # S_i_pi
        discount_factor = 0.95
        static_score = 1
        path_score = 0
        arrive_time = 0
        prev_pos = self.pos
        for i in range(len(path)):
            task = self.tasks[path[i], :]
            travel_time = self.calc_arrive_time(prev_pos, task)
            arrive_time += travel_time
            path_score += discount_factor**arrive_time * static_score
            prev_pos = task
        return path_score

    def calc_arrive_time(self, prev_pos, pos):
        dist = np.linalg.norm(pos - prev_pos, 2)
        return dist / self.vel






