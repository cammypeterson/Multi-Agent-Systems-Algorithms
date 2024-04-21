Open either file starting with "experiment_" to run one of the two scenarios.

astar.py - implementation of the astar algorithm
generator.py - generates different meshes, used to discretize the path of an obstacle
mesh.py      - has all the classes and functions relating to 3D meshes including GJK collision algorithm, mesh graphs, meshes, meshops (operations on meshes)
obj_parser.py - parses .obj files into mesh objects
optimizer4pt - optimizer code using scipy optimize.  TODO:: add bezier spline optimization functionality.