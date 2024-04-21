import numpy as np
from obj_parser import ObjParser
from mesh import Mesh, MeshOp
from copy import copy, deepcopy

class Generator:
    ''' BUILDS DIFFERENT MESHES USING BASEMESHES AND MATH'''
    def GeneratePlane(x_bounds, y_bounds, z_offset):
        delta_x = x_bounds[1] - x_bounds[0]
        delta_y = y_bounds[1] - y_bounds[0]
      
        m = ObjParser.Parse("meshes/basemesh/plane.obj")
        m.verts[:,0] *= delta_x
        m.verts[:,1] *= delta_y
        m.verts[:,2] += z_offset
        m.verts[:,0] += x_bounds[0]
        m.verts[:,1] += y_bounds[0]
        m.name = "plane_" + str(Mesh.id)
        return m
    
    def GenerateBounds(x_bounds, y_bounds, z_bounds):
        delta_x = x_bounds[1] - x_bounds[0]
        delta_y = y_bounds[1] - y_bounds[0]
        delta_z = z_bounds[1] - z_bounds[0]
        m = ObjParser.Parse("meshes/basemesh/cube.obj")
        m.verts[:,0] *= delta_x
        m.verts[:,1] *= delta_y
        m.verts[:,2] *= delta_z
        m.verts[:,0] += x_bounds[0]
        m.verts[:,1] += y_bounds[0]
        m.verts[:,2] += z_bounds[0]
        m.name = "cube_" + str(Mesh.id)
        return m

    def SegmentFromPoints(points, width):
        base_mesh = np.copy(Mesh.SQUARE_BASEMESH) * width
        new_verts    = np.empty((len(base_mesh) * (len(points))  , 3))
        new_normals  = np.empty((len(base_mesh) * (len(points)-1)+2, 3))
        new_faces       = []
        m = Mesh()

        for i in range(len(points)):
            new_verts[i*len(base_mesh):(i+1)*len(base_mesh)] = np.copy(base_mesh) + points[i]
            if i > 0:
                for j in range(len(base_mesh)):
                    k = j + 1
                    if k == len(base_mesh):
                        k = 0
                    new_face = [(i-1)*len(base_mesh)+j, (i-1)*len(base_mesh)+k, (i)*len(base_mesh)+k, (i)*len(base_mesh)+j, (i-1)*len(base_mesh)+j]
                    ab = new_verts[(i)*len(base_mesh)+j, :] - new_verts[(i-1)*len(base_mesh)+j, :]
                    bc = new_verts[(i)*len(base_mesh)+k, :] - new_verts[(i)*len(base_mesh)+j, :]
                    new_norm  = np.cross(ab,bc)
                    new_normals[(i-1)*len(base_mesh)+j, :] = new_norm / np.linalg.norm(new_norm)
                    new_faces.append(new_face)
        new_normals[-1,:] = np.array([0.,0.,1.])
        new_normals[-2,:] = np.array([0.,0.,-1.])
        bottom_face = [i for i in range(len(base_mesh)-1, -1,-1)] + [len(new_normals)-2]
        top_face = [i+len(base_mesh) * (len(points)-1) for i in range(len(base_mesh))] + [len(new_normals)-1]

        m.verts   = np.copy(new_verts)
        m.normals = np.copy(new_normals)
        m.faces   = deepcopy(new_faces + [bottom_face, top_face])
        m.name    = "track_" + str(Mesh.track_id)
        # m.flip_normals()
        Mesh.track_id += 1
        m.RemoveDuplicates()
        return m
    
    def GenerateTubeFromPoints(points, width):
        meshes = []
        for i in range(len(points)-1):
            j = i + 1
            meshes.append(Generator.SegmentFromPoints(np.concatenate([[points[i,:]], [points[j,:]]], axis=0), width))
        return meshes

    def GenerateCrossSection(meshes, x_bound, y_bound, z, export=False):
        plane = Generator.GeneratePlane(x_bound, y_bound, z)
        new_meshes = []
        for ma in meshes:
            a = MeshOp.ConvexCuts(plane, ma, flip_backface=False)
            if len(a) == 0:
                continue

            filename =  "meshes/output/cross_section_" + str(a[0].name) + ".obj"
            MeshOp.Export(a[0], filename)
            new_meshes += [a[0]]
        return new_meshes 