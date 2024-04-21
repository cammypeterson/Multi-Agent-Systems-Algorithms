from mesh import Mesh
import numpy as np
class ObjParser:
    ''' LOADS AN OBJ FILE, CONVERTS TO MESH OBJECT'''
    def __init__(self, filename) -> None:
        pass
    def Parse(filename) -> Mesh:
        m = Mesh()
        name = ""
        parser_state = ""
        verts = []
        normals = []
        faces = []
        f = open(filename, "r")
        for x in f:
            line = x[:-1]
            input_ptr = 0
            line_values = line.split(' ')
            for token in line_values:
                if token == "o":
                    parser_state = "o"
                    continue
                if parser_state == "o":
                    name = token

                if token == "v":
                    parser_state = "v"
                    verts.append(np.array([0.,0.,0.]))
                    continue
                if parser_state == "v":
                    verts[-1][input_ptr] = float(token)
                    input_ptr += 1
                
                if token == "vn":
                    parser_state = "vn"
                    normals.append(np.array([0.,0.,0.]))
                    continue
                if parser_state == "vn":
                    normals[-1][input_ptr] = float(token)
                    input_ptr += 1

                if token == "f":
                    parser_state = "f"
                    faces.append([])
                    # normals.append(np.array([0.,0.,0.]))
                    continue
                if parser_state == "f":
                    val = token.split('/')
                    faces[-1].append(int(val[0]) - 1)
                    if token == line_values[-1]:
                        faces[-1].append(int(val[-1]) - 1)
            parser_state = None

        f.close()

        verts = np.concatenate(verts).reshape((-1,3))
        normals = np.concatenate(normals).reshape((-1,3))
        m = Mesh()
        m.name = name
        m.verts = verts
        m.normals = normals
        m.faces = faces
        return m