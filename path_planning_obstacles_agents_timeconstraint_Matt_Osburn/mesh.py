import numpy as np
from copy import copy, deepcopy
import time

class MeshGraph:
    def __init__(self, meshlist):
        # self.nodes = []
        self.edges = []
        self.BuildGraph(mesh_list=meshlist)
        self.meshlist = meshlist

    def BuildGraph(self, mesh_list):
        print("Building Mesh Graph...")
        st = time.time()
        for m1 in range(len(mesh_list)-1):
            # print(f"testing {m1}")
            for mi in range(m1+1,len(mesh_list)):
                test = False
                # print(F"{mesh_list[m1].name} -> {mesh_list[mi].name}")
                gjk = GJK()
                test = gjk.test(mesh_list[m1], mesh_list[mi], inflate=1e-3)
                if test == True:
                    self.edges.append((m1, mi))
        
        reverse_list = []

        for e in self.edges:
            reverse_list.append((e[1], e[0]))
        self.edges += reverse_list

        self.edges = np.array(self.edges)

        print(f"Nodes: {len(mesh_list)}, Edges: {len(self.edges)} connected in {time.time()-st} seconds")
            

    


class Mesh:
    id = 0
    track_id = 0
    SQUARE_BASEMESH = np.array([[-0.5, -0.5, 0.],
                                [-0.5,  0.5, 0.],
                                [ 0.5,  0.5, 0.],
                                [ 0.5, -0.5, 0.]])
    def __init__(self) -> None:
        self.name = "mesh_"+str(Mesh.id)
        self.verts   = None
        self.normals = None
        self.faces   = []
        self.id = Mesh.id
        Mesh.id+=1
    def flip_normals(self):
        for i in range(len(self.faces)):
            f = self.faces[i][:-1]
            n = self.faces[i][-1]
            self.faces[i] = f[::-1] + [n]
        self.normals = -self.normals

    def get_mesh_center(self):
        return np.mean(self.verts, axis=0)

    def get_face_center(self, face):
        return np.mean(self.verts[face[:-1]], axis=0)

    ## IMPLEMENT SUPPORT VECTOR FUNCTION
    def support_vec(self, direction):
        max_vec = None
        max_dist = -np.inf
        # center = self.get_mesh_center()
        for i in range(len(self.verts)):
            dist = np.dot(self.verts[i], direction)
            if dist > max_dist:
                max_dist = dist
                max_vec = self.verts[i]
        
        return max_vec



    def GetFace(self, faceIndex):
        # numVerts = len(self.faces[faceIndex])-1
        verts = np.copy(self.verts[self.faces[faceIndex][:-1]])
        normal = np.copy(self.normals[self.faces[faceIndex][-1]])
        # order = np.arange(len(self.faces[faceIndex])-1)
        return verts, normal

    def AddFace(self, verts, normal):
        if self.verts is not None and verts is not None:
            start = len(self.verts)
            self.verts = np.concatenate([self.verts, np.copy(verts)], axis=0)
            end = len(self.verts)
            vertsPtr = [*range(start, end)]
            vertsPtr.append(np.size(self.normals,axis=0))
            self.normals = np.concatenate([self.normals, [np.copy(normal)]], axis=0)
            self.faces.append(vertsPtr)
            # self.FixLastFaceForDuplicates(self.faces[-1])
        else:
            if verts is None:
                return
            start = 0
            self.verts = np.copy(verts)
            end = len(self.verts)
            vertsPtr = [*range(start, end)]
            vertsPtr.append(0)
            self.normals = np.copy(normal)
            self.normals = np.array([self.normals])
            self.faces = [vertsPtr]


    def FixLastFaceForDuplicates(self, face):
        verts_to_remove = []
        i = 0
        while i < len(face[:-1]):
            duplicates = np.isclose(self.verts[face[i]], self.verts).all(axis=1) 
            if np.sum(duplicates) > 1:
                old_index = face[i]
                indecies = np.argwhere(duplicates)
                new_index = np.argwhere(duplicates)[0].item(0)
                face[i] = new_index
                self.verts = np.delete(self.verts,(indecies[1:]),axis=0)
                for j in range(len(face[:-1])):
                    if i ==j:
                        continue
                    if face[j] > old_index:
                        face[j]-=1
            i+=1
        
        self.verts = np.delete(self.verts, verts_to_remove, axis=0)

    def RemoveDuplicates(self):
        if self.verts is None:
            return
        if len(self.verts) < 3:
            return
        i = 0
        
        while i < len(self.verts):
            duplicates = np.argwhere(np.isclose(self.verts[i], self.verts).all(axis=1))
            if np.sum(duplicates) > 1:
                for f in self.faces:
                    # for j in range(1, len(duplicates)):
                    inds = np.argwhere(np.sum(f[:-1]==duplicates[1:], axis=0)).T[0]
                    if len(inds) ==0:
                        continue
                    else:
                        for j in inds:
                            f[j] = duplicates[0,0]
            i += 1
        currentMax = 0
        for f in self.faces:
            currentMax = np.max([currentMax,np.max(f[:-1])])

        self.verts = self.verts[:currentMax+1]

    def GetPointInMesh(self, point):  
        projections = np.empty(len(self.faces))
        for f in range(len(self.faces)):
            a = point - self.verts[self.faces[f][0]]
            proj = np.dot(self.normals[self.faces[f][-1]], a)
            projections[f] = proj
        return np.min(projections)  # Incidentally this also calculates the minimum distance from the point to the object

    def get_dist_to_Mesh(self, point):
        dists_faces = np.empty(len(self.faces))
        for f in range(len(self.faces)):
            a = point - self.get_face_center(self.faces[f]) 
            proj = np.dot(self.normals[self.faces[f][-1]], a)
            dists_faces[f] = proj
        dists_verts = np.empty(len(self.verts))
        for v in range(len(self.verts)):
            dists_verts[v] = np.linalg.norm(point - self.verts[v])
        
        if np.sum(dists_faces < 0) > 1:
            return np.min(dists_verts)
        else:
            return -np.min(dists_faces)
        
    def get_closest_point_on_mesh(self, point):
        dists_faces = np.zeros((len(self.faces),))
        for f in range(len(self.faces)):
            a = point - self.get_face_center(self.faces[f]) 
            proj = np.dot(self.normals[self.faces[f][-1]], a)
            dists_faces[f] = proj
        dists_verts = np.zeros((len(self.verts),))
        for v in range(len(self.verts)):
            dists_verts[v] = np.linalg.norm(point - self.verts[v])
        
        if np.sum(dists_faces < 0) > 1:
            retval = self.verts[np.argwhere(dists_verts - np.min(dists_verts)  == 0.).T[0]]
            if retval.ndim > 1:
                retval = retval[0, :]
            return retval
        elif np.sum(dists_faces < 0) == 1:
            f = np.argwhere(dists_faces<0).T[0].item(0)
            face_normal = self.normals[self.faces[f][-1]]
            a = point - self.get_face_center(self.faces[f]) 
            return point - np.dot(a, face_normal)*face_normal 
        else:
            return point
    # def PackageFace(self, face):
    #     numVerts = len(face)-1
    #     verts = self.verts[face[:-1]]
    #     normal = self.normals[face[-1]]
    #     order = np.arange(len(face)-1)
    #     return verts, normal, order
class GJK:

    def Inflate(a:Mesh, scale=1e-6):
        ai = Mesh()
        ai.verts = np.copy(a.verts)
        ai.faces = deepcopy(a.faces)
        ai.normals = np.copy(a.normals)
        ai_center = ai.get_mesh_center()
        ai.verts = ((ai.verts - ai_center) * (1 + scale)) + ai_center
        return ai

    def __init__(self) -> None:
        self.points = []
        self.direction = np.array([0.,0.,0.])    

    def support2(a:Mesh, b:Mesh, direction):
        return a.support_vec(direction) - b.support_vec(-direction)

    def test(self, a_:Mesh, b_:Mesh, inflate=0.):
        a = GJK.Inflate(a_, inflate)
        b = GJK.Inflate(b_, inflate)
        # a = a_
        # b = b_
        self.direction = a.get_mesh_center() - b.get_mesh_center()
        support = GJK.support2(a, b, self.direction)
        self.points = [np.copy(support)]
        self.direction = -support

        while True:
            if len(self.points) == len(a.verts) + len(b.verts):
                return False
            support = GJK.support2(a, b, self.direction)

            test = (np.dot(support, self.direction))
            # if test == 0:
            #     return True
            if test <= 0:
                return False
            
            self.points.insert(0, support)
            
            if (self.NextSimplex()):
                return True
            

    def NextSimplex(self):
        numPoints = len(self.points)
        if numPoints == 2: 
            return self.Line()
        if numPoints == 3:
            return self.Triangle()
        if numPoints == 4:
            return self.Tetrahedron()
        return False

    def SameDir(direction, ao):
        return np.dot(direction, ao) > 0

    def Line(self):
        retval = False
        a = self.points[0]
        b = self.points[1]

        ab = b - a
        ao =   - a

        if GJK.SameDir(ab, ao):
            self.direction = np.cross(np.cross(ab,ao), ab)
            
        else:
            self.points = [a]
            self.direction = ao
            
        return retval

    def Triangle(self, face_check=False):
        retval = face_check
        a = self.points[0]
        b = self.points[1]
        c = self.points[2]

        ab = b - a
        ac = c - a
        ao =   - a

        abc = np.cross(ab, ac)
        if GJK.SameDir(np.cross(abc, ac), ao):
            if GJK.SameDir(ac,ao):
                self.points = [a,c]
                self.direction = np.cross(np.cross(ac,ao),ac)

            else:
                self.points = [a,b]
                return self.Line()
        
        else:
            if GJK.SameDir(np.cross(ab,abc), ao):
                self.points = [a,b]
                return self.Line()
            else:
                if GJK.SameDir(abc, ao):
                    self.direction = abc
                    
                else:
                    self.points = [a,c,b]
                    self.direction = -abc
                    

        return retval
    
    def Tetrahedron(self):
        
        a = self.points[0]
        b = self.points[1]
        c = self.points[2]
        d = self.points[3]

        ab = b - a
        ac = c - a
        ad = d - a
        ao =   - a

        abc = np.cross(ab, ac)
        acd = np.cross(ac, ad)
        adb = np.cross(ad, ab)

        if (GJK.SameDir(abc, ao)):
            self.points = [a,b,c]
            return self.Triangle()
        
        if (GJK.SameDir(acd, ao)):
            self.points = [a,c,d]
            return self.Triangle()
        
        if (GJK.SameDir(adb, ao)):
            self.points = [a,d,b]
            return self.Triangle()
        
        return True



class MeshOp:
    

    def Export(mesh:Mesh, filename:str):
        if mesh.verts is None:
            print(f"ERROR EXPORTING TO {filename}")
            return
        f = open(filename, "w")
        lines = []
        lines.append(f"o {mesh.name}\n")
        for v in mesh.verts:
            lines.append(f"v {v[0]} {v[1]} {v[2]}\n")
        for n in mesh.normals:
            lines.append(f"vn {n[0]} {n[1]} {n[2]}\n")
        lines.append(f"s 0\n")
        for face in mesh.faces:
            string = "f "
            for i in face[:-1]:
                string+=f"{i+1}//{face[-1]+1}"
                if i != face[-2]:
                    string+=" "
               
            string+="\n"
            lines.append(string)
        f.writelines(lines)
        f.close()


    def IntersectionNumDenom(p0,p1, e0, normal):
        p01 = p1 - p0
        numerator = np.dot(e0 - p0, normal)
                # print(numerator)
        denominator = np.dot(p01, normal)

        return numerator, denominator


    def FrontBackIntersectionFaces(mesh:Mesh, plane):
        e0, normal = plane
        sign = 0

        inFront = 0
        coplanar = 0
        inBehind = 0

        front = []
        back = []
        intersection = []
        coplanarInds = []
        i = 0
        for face in mesh.faces:
            for vert in range(len(face)-1):
                k0 = vert
                k1 = vert + 1
                if k1 == len(face)-1:
                    k1=0

                p0 = mesh.verts[face[k0]]
                p1 =  mesh.verts[face[k1]]
                p01 = p1 - p0
                numerator = np.dot(p0 - e0, normal)
                # print(numerator)
                denominator = np.dot(p01, normal)

                # if denominator == 0 then lines are parrallel
                denominator_is_zero = np.isclose(denominator,0.)    
                # segments are co-planar if they are parallel and the numerator is zero
                numerator_is_zero = np.isclose(numerator,0.)

                if denominator_is_zero and numerator_is_zero:
                    coplanar+= 1
                    continue

                test = np.dot((p0 - e0), normal)
                # if np.isclose(test, 0., atol=1e-3) :
                    # test= 0

                sign = np.sign(test)
                if sign >= 0:
                    inFront += 1
                if sign <= 0:
                    inBehind += 1
            
            
            if inFront>0 and inBehind == 0:
                front.append(i)
            if inBehind > 0 and inFront == 0:
                back.append(i)
            if inFront > 0 and inBehind > 0:
                intersection.append(i)

            if coplanar == len(face)-1: 
                coplanarInds.append(i)
                
            # if inFront ==0 and inBehind==0:
                # front.append(i)
                # back.append(i)
            coplanar = 0
            inFront = 0
            inBehind = 0
            i+= 1
        return front, back, intersection, coplanarInds

    def remove_duplicates(verts):
        i = 0
        while i < len(verts):
            slice = np.isclose(verts[i], verts).all(axis=1) 
            if np.sum(slice) > 1:
                verts = np.delete(verts,(np.argwhere(slice)[1:]),axis=0)
            i+=1
        return verts
    
    def sortVerts(verts, ref_normal):
        if len(verts) < 3:
            return
        angle_list = [0.]
        p0 = verts[0]
        p1 = verts[1]
        p2 = verts[2]
        p01 = p1 - p0
        p02 = p2 - p0
        for i in range(2, len(verts)):
            pi = verts[i]
            p0i = pi - p0
            angle_list.append(np.arccos(np.dot(p0i, p01) / (np.linalg.norm(p01) * np.linalg.norm(p0i))))
        inds = np.argsort(angle_list)+1
        verts[1:] = verts[inds]

        if np.dot(np.cross(p01, p02), ref_normal) < 0:
            verts = np.flip(verts, axis=0)

        return verts

    def cut(mesh_to_cut_:Mesh, plane, fill_face=True, flip_backface=False, cull_planes=True) -> list[Mesh]:
        # Should return a list of convex meshes from the cut.
        # optionally returns the node graph between convex meshes as well
        mesh_to_cut = deepcopy(mesh_to_cut_)
        if len(mesh_to_cut.faces) == 0:
            return [Mesh(), Mesh()]

        cut_point, cut_normal = plane
        new_verts = []

        frontMesh = Mesh()
        backMesh = Mesh()

        frontIndecies, backIndecies, intersectionIndecies, coplanarIndicies = MeshOp.FrontBackIntersectionFaces(mesh_to_cut, plane)
        if len(coplanarIndicies) > 0:
            return [frontMesh, mesh_to_cut]
        # overlap = []
        # for i in range(len(intersectionIndecies)):
        #     for j in range(len(coplanarIndicies)):
        #         if intersectionIndecies[i] == coplanarIndicies[j]:
        #             overlap.append(i)


        for i in frontIndecies:
            v, n = mesh_to_cut.GetFace(i)
            frontMesh.AddFace(v, n)
        frontMesh.RemoveDuplicates()
        for i in backIndecies:
            v, n = mesh_to_cut.GetFace(i)
            backMesh.AddFace(v, n)
        backMesh.RemoveDuplicates()
        # for i in coplanarIndicies:
        #     v, n = mesh_to_cut.GetFace(i)
        #     backMesh.AddFace(v, n)
        #     frontMesh.AddFace(np.flip(v, axis=0), -n)

        verts_for_face = []
        for i in intersectionIndecies:


            coplanar = 0
            v,n = mesh_to_cut.GetFace(i)
            added_verts_face = []
            for k0 in range(len(v)): # iterate through vertices
                k1 = k0 + 1
                if k1 == len(v): # wrap around to complete face
                    k1 = 0
                
                p0 = v[k0]
                p1 = v[k1]
                p01 = p1 - p0
                numerator = np.dot(cut_point - p0, cut_normal)
                # print(numerator)
                denominator = np.dot(p01, cut_normal)
                # if denominator == 0 then lines are parrallel
                denominator_is_zero = np.isclose(denominator,0.)    
                # segments are co-planar if they are parallel and the numerator is zero
                numerator_is_zero = np.isclose(numerator,0.)
                if denominator_is_zero and numerator_is_zero:
                    coplanar+=1
                    continue

                if not denominator_is_zero:
                    t = numerator / denominator
                    if 0.0 < t < 1.0:
                        intersection_point = p0 + t*p01
                        if k1 == 0:
                            k1 = len(v)
                        added_verts_face.append((np.copy(intersection_point), [k0,k1, np.sign(numerator)]))
                        verts_for_face.append(np.copy(intersection_point))
                

            if len(added_verts_face) > 1:
                f1 = np.concatenate([np.copy(v[[*range(0, added_verts_face[0][1][0]+1)]]),
                                    np.copy([added_verts_face[0][0]]), 
                                    np.copy([added_verts_face[1][0]]), 
                                    np.copy(v[[*range(added_verts_face[1][1][1], len(v))]])])
                
                f2 = np.concatenate([np.copy(v[[*range(added_verts_face[0][1][1], added_verts_face[1][1][0]+1)]]),
                                        np.copy([added_verts_face[1][0]]),
                                        np.copy([added_verts_face[0][0]])])

                if added_verts_face[0][1][2] != -1: # #CHECK THIS
                    f1,f2 = f2,f1
                
                frontMesh.AddFace(f1, n)
                backMesh.AddFace(f2, n)

                frontMesh.RemoveDuplicates()
                backMesh.RemoveDuplicates()
    
        if len(verts_for_face) > 2 and fill_face == True:
            # TODO FIX FACE NORMALS
            verts_for_face = np.concatenate(verts_for_face).reshape((-1,3))
            verts_for_face = MeshOp.remove_duplicates(verts_for_face)

            if len(verts_for_face) > 2:
                
                verts_for_face = MeshOp.sortVerts(verts_for_face, cut_normal)
                # new_face_norm = np.cross(verts_for_face[1] - verts_for_face[0], verts_for_face[2] - verts_for_face[1])
                # if np.dot(new_face_norm, cut_normal) > 0:
                #     new_face_norm *= -1
                # new_face_norm = new_face_norm/np.linalg.norm(new_face_norm)
                
                new_face_norm = cut_normal
                if flip_backface:
                    verts_for_face = np.flip(verts_for_face, axis=0)
                    
                # frontMesh.AddFace(np.flip(verts_for_face,axis=0), -new_face_norm)
                frontMesh.AddFace(verts_for_face, new_face_norm)
                backMesh.AddFace(np.flip(verts_for_face, axis=0), -new_face_norm)

        frontMesh.RemoveDuplicates()
        backMesh.RemoveDuplicates()
        if cull_planes:
            if len(frontMesh.faces) < 4:
                frontMesh = Mesh()
            if len(backMesh.faces) < 4:
                backMesh = Mesh()

        

        return [frontMesh, backMesh]


    def CutMesh(mesh:Mesh, cutters:list[Mesh]):
        print(f"Cutting Mesh with {len(cutters)} cutters...")
        st = time.time()
        meshes = [mesh] 
        while len(cutters) > 0:
            temp = []
            for i in range(len(meshes)):
                gjk = GJK()
                test = gjk.test(meshes[i], cutters[0], -1e-6)
                if test == True:
                    fl, _ = MeshOp.ConvexCut(meshes[i], cutters[0])
                    temp += fl
                else:
                    temp += [meshes[i]]
            cutters = cutters[1:]
            meshes = temp
        print(f"Subdivided into {len(meshes)} segments in {time.time()-st} seconds")
        return meshes


    def ConvexCut(mesh_: Mesh, cutterMesh_: Mesh, flip_backface=False, fill_backface = True):
        mesh       = deepcopy(mesh_)
        cutterMesh = deepcopy(cutterMesh_)
        frontList  = []
        backMesh   = mesh

        for i in range(len(cutterMesh.faces)): #range(len(cutterMesh.faces)):
            face = Mesh()
            v = cutterMesh.verts[cutterMesh.faces[i][:-1]]
            n = cutterMesh.normals[cutterMesh.faces[i][-1]]
            face.AddFace(v, n)
            gjk = GJK()
            if gjk.test(mesh, face):
                cut_point = cutterMesh.get_face_center(cutterMesh.faces[i]) #(cutterMesh.verts[cutterMesh.faces[i][0]] + cutterMesh.verts[cutterMesh.faces[i][1]])/2
                normal    = np.copy(cutterMesh.normals[cutterMesh.faces[i][-1]])
                cut_point += normal*1e-3
                f,  b   = MeshOp.cut(backMesh, (cut_point, normal), flip_backface=flip_backface, fill_face=fill_backface)
                if f.verts is not None:
                    frontList.append(f)
       
                backMesh = b
            else:
                continue


        return frontList, [backMesh]

    # def ConvexCuts(mesh_: Mesh, cutterMesh_: Mesh, flip_backface=False, fill_backface = True, get_backspace = False):
    #     ''' FIX NOTATION :O :O :O, Front and back notation not accurate, but it just works'''
    #     mesh = deepcopy(mesh_)
    #     cutterMesh = deepcopy(cutterMesh_)
    #     frontList = []
    #     backList  = []
    #     fl = []
    #     bl = []
    #     # if mesh is None:
    #     #     return []
    #     # if len(mesh.faces) ==0 :
    #     #     return []
    #     # if cutterMesh is None:
    #     #     return [mesh]
    #     # if len(cutterMesh.faces) == 0:
    #     #     return [mesh]

    #     cut_point = np.copy(cutterMesh.verts[cutterMesh.faces[0][0]])
    #     normal    = np.copy(cutterMesh.normals[cutterMesh.faces[0][-1]])

    #     cutterMesh.faces = cutterMesh.faces[1:]
    #     frontIndecies, backIndecies, intersectionIndecies = MeshOp.FrontBackIntersectionFaces(cutterMesh, (cut_point, normal))
    #     fc = Mesh()
    #     bc = Mesh()
    #     for i in frontIndecies:
    #         fc.AddFace(cutterMesh.verts[cutterMesh.faces[i][:-1]], cutterMesh.normals[cutterMesh.faces[i][-1]])
    #     for i in backIndecies:
    #         bc.AddFace(cutterMesh.verts[cutterMesh.faces[i][:-1]], cutterMesh.normals[cutterMesh.faces[i][-1]])
    #     for i in intersectionIndecies:
    #         fc.AddFace(cutterMesh.verts[cutterMesh.faces[i][:-1]], cutterMesh.normals[cutterMesh.faces[i][-1]])
    #         bc.AddFace(cutterMesh.verts[cutterMesh.faces[i][:-1]], cutterMesh.normals[cutterMesh.faces[i][-1]])

    #     f,  b   = MeshOp.cut(mesh, (cut_point, normal), flip_backface=flip_backface, fill_face=fill_backface)
        
    #     if len(fc.faces) > 0:
    #         if len(f.faces)>0:
    #             fl = MeshOp.ConvexCuts(f, fc, get_backspace=get_backspace)
    #     else:
    #         if len(f.faces)>0:
    #             if get_backspace==False:
    #                 fl+= [f]
        
    #     if len(bc.faces) > 0:
    #         if len(b.faces)>0:
    #             bl = MeshOp.ConvexCuts(b, bc, get_backspace=get_backspace)
    #     else:
    #         if len(b.faces)>0:
    #             if get_backspace==True:
    #                 bl+= [b]
                

    #     frontList += fl
    #     backList += bl

    #     return backList + frontList

    

    def UnionMeshes(meshes:list[Mesh]):
        new_mesh = Mesh()
        new_mesh.verts   = np.concatenate([m.verts for m in meshes], axis=0)
        new_mesh.normals = np.concatenate([m.normals for m in meshes], axis=0)
        num_verts = 0
        num_normals = 0
        new_mesh.faces = deepcopy(meshes[0].faces)
        for i in range(1, len(meshes)):
            num_verts   += len(meshes[i-1].verts)
            num_normals += len(meshes[i-1].normals)
            faces = deepcopy(meshes[i].faces)
            for f in faces:
                for j in range(len(f[:-1])):
                    f[j] += num_verts
                f[-1] += num_normals

            new_mesh.faces += faces
     

        return new_mesh


    def MultiCutters(mesh:Mesh, cutters:list[Mesh]):
        working_set = [mesh]
        while len(cutters) > 0:
            temp = []
            for m in working_set:
                gjk = GJK()
                test = gjk.test(m, cutters[0])
                if test == True:
                    new_meshes = MeshOp.ConvexCuts(m, cutters[0], flip_backface=False, fill_backface=True, get_backspace=False)
                    temp += new_meshes

                    # filename  =  "meshes/output/" + str(new_meshes[0].name) + ".obj"
                    # MeshOp.Export(new_meshes[0], filename)
            working_set = temp
            cutters = cutters[1:]
        return working_set
