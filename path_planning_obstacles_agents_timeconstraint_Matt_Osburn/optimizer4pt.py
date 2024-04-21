
import numpy as np
from scipy.optimize import minimize
import time
from mesh import Mesh, GJK
class Optimizer:
    ''' OPTIMIZATION BUILDER'''
    def __init__(self, start_point, end_point, solution_mesh_list, cutter_meshes, pointsPerMesh=2):
        self.start_point = start_point
        self.end_point = end_point
        self.solution_mesh_list = solution_mesh_list
        self.points_per_mesh = pointsPerMesh
        self.point_dim = 3
        self.max_vel = 10
        self.res = None
        self.cutter_meshes = cutter_meshes

    def cost_function(self, designVars):
        dist = 0.
        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            
            for p in range(self.points_per_mesh-1):
                j = i+self.point_dim
                k = j+self.point_dim
                a = designVars[i:j]
                b = designVars[j:k]
                dist += np.linalg.norm(b[:2] - a[:2])
                i = j
            
        return dist
    
    def constraint_bounds(self, designVars):  # ineq
        cons = np.zeros((self.points_per_mesh*len(self.solution_mesh_list),))
        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh

            for p in range(self.points_per_mesh):
                j = i+self.point_dim
                a = designVars[i:j]
                cons[self.points_per_mesh*m+p] = self.solution_mesh_list[m].GetPointInMesh(a)
                i = j

        return cons
    
    def constraint_in_safe_set(self, designVars):
        cons = np.zeros((self.points_per_mesh*len(self.solution_mesh_list),))
        for m in range(0, len(self.solution_mesh_list)):

            i = m*self.point_dim*self.points_per_mesh

            for p in range(self.points_per_mesh):
                j = i+self.point_dim
                a = designVars[i:j]
                mina = 10
                for mesh in range(len(self.cutter_meshes)):
                    mina = np.min([mina, self.cutter_meshes[mesh].GetPointInMesh(a)])

                cons[self.points_per_mesh*m+p] = -mina
                i = j

        return cons
    
    def constraint_GJK(self, designVars):
        cons = np.zeros((len(self.solution_mesh_list),))
        for m in range(0, len(self.solution_mesh_list)):
            segment_hull = Mesh()
            i = m*self.point_dim*self.points_per_mesh
            verts = []
            for p in range(self.points_per_mesh):
                j = i+self.point_dim
                a = designVars[i:j]
                verts.append(a)
                i = j
            verts = np.concatenate(verts).reshape((-1,3))
            segment_hull.verts = verts
            for cut in self.cutter_meshes:
                gjk = GJK()
                if gjk.test(segment_hull, cut):
                    cons[m] -= 0.1/np.linalg.norm(cut.get_mesh_center() - segment_hull.get_mesh_center())
                    break
                    
       

        return cons
    
    def constraint_time_increase(self, designVars): # ineq
        cons = np.zeros(((self.points_per_mesh-1)*len(self.solution_mesh_list),))

        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh

            for p in range(self.points_per_mesh-1):
                j = i+self.point_dim
                k = j+self.point_dim
                a = designVars[i:j]
                b = designVars[j:k]
                cons[(self.points_per_mesh-1)*m+p]   = (b[2] - a[2]) 
                i = j

        return cons

    def constraint_velocity(self, designVars): # ineq
        cons = np.zeros(((self.points_per_mesh-1)*len(self.solution_mesh_list),))

        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh

            for p in range(self.points_per_mesh-1):
                j = i+self.point_dim
                k = j+self.point_dim
                a = designVars[i:j]
                b = designVars[j:k]
                dT = b[2] - a[2]
                dist = np.linalg.norm(b[:2] - a[:2])
                cons[(self.points_per_mesh-1)*m+p]   = self.max_vel - dist / dT
                i = j

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
        for m in range(1, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh - self.point_dim
            j = m*self.point_dim*self.points_per_mesh 
            k = m*self.point_dim*self.points_per_mesh + self.point_dim
            b = designVars[i:j]
            c = designVars[j:k]
            cons[(m-1)*self.point_dim:(m)*self.point_dim]   = c-b
        return cons
    
    def Solve(self):
        print("Solving Optimization...")
        st = time.time()

        x0 = np.zeros((self.point_dim*self.points_per_mesh*len(self.solution_mesh_list),))
        for m in range(0, len(self.solution_mesh_list)):
            i = m*self.point_dim*self.points_per_mesh
            t_offset = 0.
            # INITIALIZE POINTS 
            for p in range(self.points_per_mesh):
                j = i+self.point_dim
                x0[i:j] = self.solution_mesh_list[m].get_mesh_center()
                x0[j-1] += t_offset
                t_offset += 0.05
                i = j
        # print(x0)
        con_bounds                  = {'type': 'ineq', 'fun': self.constraint_bounds}
        con_time_increase           = {'type': 'ineq', 'fun': self.constraint_time_increase}
        con_velocity                = {'type': 'ineq', 'fun': self.constraint_velocity}
        con_endpoints               = {'type': 'eq', 'fun':   self.constraint_endpoints}
        con_start_end_conditions    = {'type': 'eq', 'fun':   self.constraint_start_end}
        con_hulls                   = {'type': 'ineq', 'fun':   self.constraint_GJK}

        cons = [con_bounds, con_time_increase, con_velocity, con_endpoints, con_start_end_conditions]
        self.res = minimize(self.cost_function, x0, method="trust-constr", constraints=cons, options={"maxiter": 2000, "verbose":2}, tol=1e-4)
        print(f"Optimization Solved, see optimizer.res for results [{time.time()-st} seconds]")
        
    def GetPoints(self, cut_duplicates=False):
        if self.res is None:
            return None
        pts = []
        i = 0
        while i < len(self.res.x):
            mult = 1
            pts.append(self.res.x[i:i+self.point_dim])
            if cut_duplicates == True:
                if i > 0:
                    mult=2
            i+=self.point_dim*mult
        # for i in range(0,len(self.res.x), self.point_dim):
        #     pts.append(self.res.x[i:i+self.point_dim])
        #     if cut_duplicates == True:
        #         if i > 0:
        #             i += 1

        return np.concatenate(pts).reshape((-1,self.point_dim))