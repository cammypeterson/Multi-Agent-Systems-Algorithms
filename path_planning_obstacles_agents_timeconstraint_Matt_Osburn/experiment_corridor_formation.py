import numpy as np
from mesh import *
from obj_parser import ObjParser
from generator import Generator
from astar import A_Star_Time
from optimizer4pt import Optimizer
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import matplotlib.patches as patches

def getXYfromLinePath(t, points, points_per_mesh=2):
    if t < points[0,2]:
        xy = points[0, :2]
        return xy[0], xy[1]
    if t > points[-1,2]:
        xy = points[-1, :2]
        return xy[0], xy[1]
    
    i = 0
    for m in range(1, len(points), 2):
        if t > points[m, 2]:
            i+=1
        else:
            break
    r = (t - points[i*points_per_mesh,2]) / (points[i*points_per_mesh+1,2] - points[i*points_per_mesh,2])
    ab = points[i*points_per_mesh + 1,:2] - points[i*points_per_mesh,:2]
    xy = points[i*points_per_mesh,:2] + r * ab
    return xy[0], xy[1]

END_TIME = 6.

width = 0.5

agent_paths = []
agent_points = []


# start_points = [np.array([0.325,      5/3., 0.]),
#                 np.array([0.325+0.8, 5/3., 0.]),
#                 np.array([0.325,      5*2/3., 0.]),
#                 np.array([0.325+0.8, 5*2/3., 0.])]

# end_points   = [np.array([5-(0.325),           5/3., END_TIME]),
#                 np.array([5-(0.325+0.8),      5/3., END_TIME+1]),
#                 np.array([5-(0.325),           5*2/3., END_TIME+2]),
#                 np.array([5-(0.325+0.8),      5*2/3., END_TIME+3])]

start_points = [np.array([0.325,      5/3., 0.]),
                np.array([0.325+0.8, 5/3., 0.]),
                np.array([0.325,      5*2/3., 0.]),
                np.array([0.325+0.8, 5*2/3., 0.])]

end_points   = [np.array([5-(0.325),           5/3*2., END_TIME]),
                np.array([5-(0.325+0.8),      5/3*2., END_TIME]),
                np.array([5-(0.325),           5/3., END_TIME]),
                np.array([5-(0.325+0.8),      5/3., END_TIME])]


solutions = []

h0_x1 = 1.5
h0_x2 = 3.5
h0_y1 = -1.
h0_y2 = 2.5

h1_x1 = 1.5
h1_x2 = 3.5
h1_y1 = 3.5
h1_y2 = 6.

# LOOP FOR PLANNING
for i in range(4):
    bounds = Generator.GenerateBounds([0., 5.], [0., 5.],     [0.,  END_TIME+4])
    hall01 = Generator.GenerateBounds([h0_x1-width/2, h0_x2+width/2], [h0_y1-width/2, h0_y2+width/2], [-0.1, END_TIME+5])
    hall02 = Generator.GenerateBounds([h1_x1-width/2, h1_x2+width/2], [h1_y1-width/2, h1_y2+width/2], [-0.1, END_TIME+5])

    hall01.flip_normals()
    hall02.flip_normals()
    world_bounds = [hall01, hall02]
    m = MeshOp.CutMesh(bounds, world_bounds + agent_paths) # CUT BOUNDS USING AGENTS AND OBSTACLES
    mg = MeshGraph(m)  # BUILD MESH GRAPH
    a_star = A_Star_Time(mg, start_points[i], end_points[i])
    sol = a_star.Solve() # SOLVE A* problem
    solved_mesh_list = []
    for j in sol:
        solved_mesh_list.append(m[j]) # Get meshes relating to A* solution
    opt = Optimizer(start_points[i], end_points[i], solved_mesh_list, world_bounds + agent_paths, pointsPerMesh=2) # Solve path planning problem
    opt.Solve() # Solve
    solutions.append(opt) # Get solution for this agent
    new_points = opt.GetPoints(True)  # Return the actual control points for the path
    new_points = np.concatenate([new_points, [new_points[-1,:]]], axis=0)
    new_points[-1,2] = END_TIME+5  # There is an extra point that we set outside the bounds time axis bound, it makes cutting cleaner
    agent_paths += Generator.GenerateTubeFromPoints(new_points, width*2) #  Build a tube to re-cut the bounds
    agent_points.append(opt.GetPoints(False))


### GRAPHING CODE ###

t_end = END_TIME+4
t = np.linspace(0., t_end, int(t_end*30))

fig, ax = plt.subplots()
plt.gca().set_aspect('equal')

ax.scatter(start_points[0][0], start_points[0][1], c="green", marker="*")
ax.scatter(end_points[0][0], end_points[0][1],     c="green", marker="*")

ax.scatter(start_points[1][0], start_points[1][1], c="green", marker="*")
ax.scatter(end_points[1][0], end_points[1][1],     c="green", marker="*")

ax.scatter(start_points[2][0], start_points[2][1], c="green", marker="*")
ax.scatter(end_points[2][0], end_points[2][1],     c="green", marker="*")

ax.scatter(start_points[3][0], start_points[3][1], c="green", marker="*")
ax.scatter(end_points[3][0], end_points[3][1],     c="green", marker="*")

plt.xlim([0., 5.])
plt.ylim([0., 5.])

agent_scatter0 = ax.scatter(agent_points[0][0,0], agent_points[0][0,1], c="red")
agent_scatter1 = ax.scatter(agent_points[1][0,0], agent_points[1][0,1], c="red")
agent_scatter2 = ax.scatter(agent_points[2][0,0], agent_points[2][0,1], c="red")
agent_scatter3 = ax.scatter(agent_points[3][0,0], agent_points[3][0,1], c="red")

agent_box0 = patches.Rectangle((agent_points[0][0,0]-width/2, agent_points[0][0,1]-width/2) , width , width, linewidth=1.0, edgecolor='black', facecolor="none")
agent_box1 = patches.Rectangle((agent_points[1][0,0]-width/2, agent_points[1][0,1]-width/2) , width , width, linewidth=1.0, edgecolor='black', facecolor="none")
agent_box2 = patches.Rectangle((agent_points[2][0,0]-width/2, agent_points[2][0,1]-width/2) , width , width, linewidth=1.0, edgecolor='black', facecolor="none")
agent_box3 = patches.Rectangle((agent_points[3][0,0]-width/2, agent_points[3][0,1]-width/2) , width , width, linewidth=1.0, edgecolor='black', facecolor="none")


hall01_patch = patches.Rectangle((h0_x1 + width/2, h0_y1+width/2) , h0_x2-h0_x1 - width/2, h0_y2-h0_y1 - width/2, linewidth=1.0, edgecolor='black', facecolor="none")
hall02_patch = patches.Rectangle((h1_x1 + width/2, h1_y1+width/2) , h1_x2-h1_x1 - width/2, h1_y2-h1_y1 - width/2, linewidth=1.0, edgecolor='black', facecolor="none")

txt = plt.text(0.9, 0.99, 't=0.',ha='right', va='top', transform=ax.transAxes)


ax_patch_box_agent = [agent_scatter0,
                      agent_scatter1,
                      agent_scatter2,
                      agent_scatter3]

ax_patch_box_0 = ax.add_patch(hall01_patch)
ax_patch_box_1 = ax.add_patch(hall02_patch)

ax_patch_abox_0 = ax.add_patch(agent_box0)
ax_patch_abox_1 = ax.add_patch(agent_box1)
ax_patch_abox_2 = ax.add_patch(agent_box2)
ax_patch_abox_3 = ax.add_patch(agent_box3)


plt.xlabel("pos (x)")
plt.ylabel("pos (y)")
plt.title("Simulation, 4 agents")
fig.tight_layout()
    

def update(frame):
    time = t[frame]
    x_0, y_0 = getXYfromLinePath(time, agent_points[0], 2)
    x_1, y_1 = getXYfromLinePath(time, agent_points[1], 2)
    x_2, y_2 = getXYfromLinePath(time, agent_points[2], 2)
    x_3, y_3 = getXYfromLinePath(time, agent_points[3], 2)

    ax_patch_box_agent[0].set_offsets((x_0,y_0))
    ax_patch_box_agent[1].set_offsets((x_1,y_1))
    ax_patch_box_agent[2].set_offsets((x_2,y_2))
    ax_patch_box_agent[3].set_offsets((x_3,y_3))

    ax_patch_abox_0.set_xy((x_0-width/2, y_0-width/2))
    ax_patch_abox_1.set_xy((x_1-width/2, y_1-width/2))
    ax_patch_abox_2.set_xy((x_2-width/2, y_2-width/2))
    ax_patch_abox_3.set_xy((x_3-width/2, y_3-width/2))

    timefloat= np.round(time,2)
    txt.set_text(f"t={timefloat:.2f}")
    
    return (ax_patch_box_agent[0], ax_patch_box_agent[1], ax_patch_box_agent[2], ax_patch_box_agent[3], txt, ax_patch_abox_0, ax_patch_abox_1, ax_patch_abox_2, ax_patch_abox_3)


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=1000/30, blit=True,repeat_delay=0.1)
plt.show()
ani.save("_animation.gif")




plt.show()

