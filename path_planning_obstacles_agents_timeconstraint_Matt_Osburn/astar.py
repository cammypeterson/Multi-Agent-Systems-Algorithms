
import numpy as np
from mesh import Mesh
import time
class A_Star_Node:

    '''
    Houses all the node information for the A* algorithm
    '''
    def __init__(self, parent=None, mesh:Mesh=None, index=None) -> None:
        self.parent = parent
        self.position = mesh.get_mesh_center()
        self.index = index
        self.g = 0
        self.h = 0
        self.f = 0
        self.mesh = mesh
        self.closest_point = None

    def __eq__(self, other) -> bool:
        ''' Checks to see if two nodes are the same'''
        return self.index == other.index

class A_Star_Time:
    ''' A star with specialized f,g, h calculations to find approximate paths through space and time'''
    def GetMeshIndexFromPoint(self, point):
        ''' Finds the index in the mesh graph that a point is interior too'''
        for m in range(len(self.mesh_graph.meshlist)):
            if self.mesh_graph.meshlist[m].GetPointInMesh(point) >= 0.:
                return m

    def __init__(self, meshgraph, start_point, end_point) -> None:
        self.mesh_graph = meshgraph
        self.start_point = start_point
        self.end_point = end_point

        self.start_ind = self.GetMeshIndexFromPoint(start_point)
        self.end_ind = self.GetMeshIndexFromPoint(end_point)

    def Solve(self, end_fixed=True):
        print("Solving Graph...")
        st = time.time()
        start_node = A_Star_Node(None, self.mesh_graph.meshlist[self.start_ind], self.start_ind)
        start_node.position = self.start_point
        start_node.closest_point = self.mesh_graph.meshlist[self.start_ind].get_closest_point_on_mesh(self.end_point)
        end_node   = A_Star_Node(None, self.mesh_graph.meshlist[self.end_ind], self.end_ind)
        
        openList = [start_node]
        closedList = []

        while len(openList) > 0:
       
            # Get the current node
            current_node = openList[0]
            current_index = 0
            for index, item in enumerate(openList):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

                # Pop current off open list, add to closed list
            openList.pop(current_index)
            closedList.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.index)
                    current = current.parent
                print(f"Solved Graph in {time.time()-st} seconds")
                return path[::-1]

            children = []
            outgoing_inds = np.argwhere(self.mesh_graph.edges[:,0]==current_node.index).T[0]
            for i in range(len(outgoing_inds)):
                ind = self.mesh_graph.edges[outgoing_inds[i]][1]
                mesh_ = self.mesh_graph.meshlist[ind]
                new_node = A_Star_Node(current_node, mesh_, ind)
                new_node.closest_point = mesh_.get_closest_point_on_mesh(self.end_point)
                children.append(new_node)

            for child in children:
                child_in_list = False
                for closed_child in closedList:
                    if child == closed_child:
                        child_in_list = True
                        break
                if child_in_list:
                    continue
                
                # max_vel = 2.
                # g_time_cost_den = child.closest_point[2] - current_node.position[2]
                # if g_time_cost_den < 0.:
                #     g_time_cost_den = 0.
                # if np.isclose(g_time_cost_den, 0.):
                #     g_time_cost_den = 1e-6
                # g_time_cost = 1/g_time_cost_den
                # g_vel_cost = np.linalg.norm(current_node.position[:2] - child.closest_point[:2]) * g_time_cost
                # if g_vel_cost > max_vel:
                #     g_vel_cost = 1000.

                # h_time_cost_den = child.closest_point[2] - self.end_point[2]
                # if h_time_cost_den < 0.:
                #     h_time_cost_den = 0.
                # if np.isclose(h_time_cost_den, 0.):
                #     h_time_cost_den = 1e-6
                # h_time_cost = 1/h_time_cost_den
                # h_vel_cost = np.linalg.norm(self.end_point[:2] - child.closest_point[:2]) * h_time_cost
                # if h_vel_cost > max_vel:
                #     h_vel_cost = 1000

                child.g = current_node.g + np.linalg.norm(current_node.position[:2] - child.closest_point[:2])
                child.h = np.linalg.norm(self.end_point[:2] - child.closest_point[:2]) 
                child.f = child.g + child.h

                for open_node in openList:
                    if child == open_node and child.g > open_node.g:
                        continue
                
                openList.append(child)
            
            # print(len(openList))
        print("---NO SOLUTION FOUND---")
        

class A_Star:
    '''UNUSED, Sanity test A* class'''
    def GetMeshIndexFromPoint(self, point):
        for m in range(len(self.mesh_graph.meshlist)):
            if self.mesh_graph.meshlist[m].GetPointInMesh(point) >= 0.:
                return m

    def __init__(self, meshgraph, start_point, end_point) -> None:
        self.mesh_graph = meshgraph
        self.start_point = start_point
        self.end_point = end_point

        self.start_ind = self.GetMeshIndexFromPoint(start_point)
        self.end_ind = self.GetMeshIndexFromPoint(end_point)

    def Solve(self):

        start_node = A_Star_Node(None, self.mesh_graph.meshlist[self.start_ind], self.start_ind)
        start_node.position = self.start_point
        start_node.closest_point = self.mesh_graph.meshlist[self.start_ind].get_closest_point_on_mesh(self.end_point)
        end_node   = A_Star_Node(None, self.mesh_graph.meshlist[self.end_ind], self.end_ind)
        
        openList = [start_node]
        closedList = []

        while len(openList) > 0:
       
            # Get the current node
            current_node = openList[0]
            current_index = 0
            for index, item in enumerate(openList):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

                # Pop current off open list, add to closed list
            openList.pop(current_index)
            closedList.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.index)
                    current = current.parent
                return path[::-1]

            children = []
            outgoing_inds = np.argwhere(self.mesh_graph.edges[:,0]==current_node.index).T[0]
            for i in range(len(outgoing_inds)):
                ind = self.mesh_graph.edges[outgoing_inds[i]][1]
                mesh_ = self.mesh_graph.meshlist[ind]
                new_node = A_Star_Node(current_node, mesh_, ind)
                new_node.closest_point = mesh_.get_closest_point_on_mesh(self.end_point)
                children.append(new_node)

            for child in children:
                for closed_child in closedList:
                    if child == closed_child:
                        continue
                        
                # g_time_cost_den = child.position[3] - current_node.position[3]
                # if g_time_cost_den < 0.:
                #     g_time_cost_den = 0.
                # if np.isclose(g_time_cost_den, 0.):
                #     g_time_cost_den = 1e-6
                # g_time_cost = 1/g_time_cost_den

                # h_time_cost_den = child.position[3] - self.end_point[3]
                # if h_time_cost_den < 0.:
                #     h_time_cost_den = 0.
                # if np.isclose(h_time_cost_den, 0.):
                #     h_time_cost_den = 1e-6
                # h_time_cost = 1/h_time_cost_den

                child.g = current_node.g + np.linalg.norm(current_node.closest_point - child.closest_point)
                child.h = np.linalg.norm(child.closest_point - self.end_point) 
                child.f = child.g + child.h

                for open_node in openList:
                    if child == open_node and child.g > open_node.g:
                        continue
                
                openList.append(child)