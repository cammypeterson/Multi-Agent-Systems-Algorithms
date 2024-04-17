
import numpy as np
from scipy.optimize import minimize
import time
class Optimizer:
    def __init__(self, start_point, end_point, solution_mesh_list, cutter_meshes):
        self.start_point = start_point
        self.end_point = end_point
        self.solution_mesh_list = solution_mesh_list
        self.points_per_mesh = 2
        self.point_dim = 3
        self.max_vel = 4
        self.res = None
        self.cutter_meshes = cutter_meshes

    def cost_function(self, designVars):
        dist = 0.
        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            j = i+self.point_dim
            k = j+self.point_dim
            a = designVars[i:j]
            b = designVars[j:k]
            c = b - a
            dist += np.linalg.norm(c[:2])
        return dist
    
    def constraint_bounds(self, designVars):  # ineq
        cons = np.zeros((self.points_per_mesh*len(self.solution_mesh_list),))
        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            j = i+self.point_dim
            k = j+self.point_dim
            a = designVars[i:j]
            b = designVars[j:k]
            cons[2*m]   = self.solution_mesh_list[m].GetPointInMesh(a)
            cons[2*m+1] = self.solution_mesh_list[m].GetPointInMesh(b)
        return cons
    
    def constraint_in_safe_set(self, designVars):
        cons = np.zeros((self.points_per_mesh*len(self.solution_mesh_list),))
        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            j = i+self.point_dim
            k = j+self.point_dim
            a = designVars[i:j]
            b = designVars[j:k]

            mina = 10
            minb = 10
            for mesh in range(len(self.cutter_meshes)):
                mina = np.min([mina, self.cutter_meshes[mesh].GetPointInMesh(a)])
                minb = np.min([minb, self.cutter_meshes[mesh].GetPointInMesh(b)])
            cons[2*m]   = -mina
            cons[2*m+1] = -minb
        return cons
    
    def constraint_time_increase(self, designVars): # ineq
        cons = np.zeros(((self.points_per_mesh-1)*len(self.solution_mesh_list),))

        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            j = i+self.point_dim
            k = j+self.point_dim
            a = designVars[i:j]
            b = designVars[j:k]
            cons[m]   = b[2] - a[2]
        return cons

    def constraint_velocity(self, designVars): # ineq
        cons = np.zeros(((self.points_per_mesh-1)*len(self.solution_mesh_list),))

        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            j = i+self.point_dim
            k = j+self.point_dim
            a = designVars[i:j]
            b = designVars[j:k]
            dT = b[2] - a[2]
            dist = np.linalg.norm(b[:2] - a[:2])
            # if np.isclose(dT, 0.):
            #     dT = -1e-3
            cons[m]   = self.max_vel - dist / dT
        return cons

    def constraint_start_end(self, designVars): # eq
        cons = np.zeros((2*self.point_dim,))
        a = designVars[0:self.point_dim]
        b = designVars[-self.point_dim:]
        cons[:self.point_dim] = self.start_point - a
        cons[self.point_dim:] = self.end_point   - b
        return cons
    
    def constraint_endpoints(self, designVars): # eq
        if len(self.solution_mesh_list) == 1:
            return np.array([0.])
        cons = np.zeros(((len(self.solution_mesh_list)-1)*self.point_dim,))
        for m in range(0, len(self.solution_mesh_list)-1):
            i = m*self.point_dim*self.points_per_mesh
            j = i+self.point_dim
            k = j+self.point_dim
            l = k+self.point_dim

            b = designVars[j:k]
            c = designVars[k:l]
            cons[m*self.point_dim:(m+1)*self.point_dim]   = c-b
        return cons
    
    def Solve(self):
        print("Solving Optimization...")
        st = time.time()

        x0 = np.zeros((self.point_dim*self.points_per_mesh*len(self.solution_mesh_list),))
        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            j = i+self.point_dim
            k = j+self.point_dim
            x0[i:j] = self.solution_mesh_list[m].get_mesh_center()
            x0[j:k] = self.solution_mesh_list[m].get_mesh_center()
            x0[k-1] += 0.1
        # print(x0)
        con_bounds                  = {'type': 'ineq', 'fun': self.constraint_in_safe_set}
        con_time_increase           = {'type': 'ineq', 'fun': self.constraint_time_increase}
        con_velocity                = {'type': 'ineq', 'fun': self.constraint_velocity}
        con_endpoints               = {'type': 'eq', 'fun':   self.constraint_endpoints}
        con_start_end_conditions    = {'type': 'eq', 'fun':   self.constraint_start_end}

        cons = [con_bounds, con_time_increase, con_velocity, con_endpoints, con_start_end_conditions]
        self.res = minimize(self.cost_function, x0, method="trust-constr", constraints=cons, options={"maxiter": 1000, "verbose":2}, tol=1e-4)
        print(f"Optimization Solved, see optimizer.res for results [{time.time()-st} seconds]")
        
    def GetPoints(self):
        if self.res is None:
            return None
        pts = []
        for i in range(0,len(self.res.x), self.point_dim):
            pts.append(self.res.x[i:i+self.point_dim])

        return np.concatenate(pts).reshape((-1,self.point_dim))