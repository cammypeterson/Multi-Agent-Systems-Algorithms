import numpy as np
from mesh import *
from obj_parser import ObjParser
from generator import Generator
from astar import A_Star_Time
from optimizer4pt import Optimizer
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import matplotlib.patches as patches

''' FOLLOWS SAME FLOW AS experiment_corridor_formation.py'''



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

def AugmentPoints(points):
    start = np.array([points[0,:]])
    end = np.array([points[-1, :]])
    middle = np.repeat(points[1:-1, :], 2, axis=0)
    return np.concatenate([start, middle, end], axis=0)

width = 0.1

NUM_POINTS = 2
END_TIME = 6.

points0 = np.array([[1.,1.,0.],
                    [1.0, 0.1, 1.0],
                    [1.5, 1.5, 4.],
                    [0.3, 1., 6.]])

points1 = np.array([[1.,1.,0.],
                    [2.0, 1.5, 3.0],
                    [3., 2., 4.],
                    [4., 4., 5.],
                    [4., 4., 6.]])

points2 = np.array([[1.,1.,0.],
                    [2.0, 0.5, 1.0],
                    [1., 2., 3.],
                    [0.3, 0.5, 4.],
                    [0.3, 0.5, 6.]]) + np.array([1.2, 1.1, 0.])

middle = [Generator.GenerateBounds([1.5, 2.5], [1.8, 2.2], [-0.1, END_TIME+0.1])]
middle[0].flip_normals()

obstacle_width = width*4

m0 = Generator.GenerateTubeFromPoints(points0, obstacle_width)
m1 = Generator.GenerateTubeFromPoints(points1, obstacle_width)
m2 = Generator.GenerateTubeFromPoints(points2, obstacle_width)
bounds = Generator.GenerateBounds([0., 4.], [0., 4.], [0., END_TIME])

m = MeshOp.CutMesh(bounds, middle + m0 + m1 + m2) 

for a in m:
    MeshOp.Export(a, "meshes/output/"+a.name+".obj")


mg = MeshGraph(m)

start_point = np.array([2., 0., 0.])
end_point   = np.array([ 2.0, 3.5, END_TIME])

a_star = A_Star_Time(mg, start_point, end_point)
sol = a_star.Solve()

print(sol)
for a in sol:
    print(m[a].name)
solved_mesh_list = []
for i in sol:
    solved_mesh_list.append(m[i])

opt = Optimizer(start_point, end_point, solved_mesh_list, middle+ m0 + m1 + m2, pointsPerMesh=NUM_POINTS)
opt.Solve()

print("Success = ", opt.res.success)
print(opt.constraint_GJK(opt.res.x))
points = opt.GetPoints()

print("num_points = ", len(points))
print(points)

plt.figure()
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.scatter(start_point[0], start_point[1], c="green", marker="*")
plt.scatter(end_point[0], end_point[1],     c="green", marker="*")
for i in range(0, len(points)-1):
    a = points[i,:]
    b = points[i+1,:]
    plt.plot([a[0], b[0]], [a[1], b[1]])
    plt.scatter(a[0], a[1], c="red", s= 20)
    plt.scatter(b[0], b[1], c="red", s= 20)
plt.show()

plt.figure()
plt.xlim([0, 6])
plt.ylim([0, 4])
plt.scatter(start_point[2], start_point[0], c="green", marker="*")
plt.scatter(end_point[2], end_point[0],     c="green", marker="*")
for i in range(0, len(points)-1):
    a = points[i,:]
    b = points[i+1,:]
    plt.plot([a[2], b[2]], [a[0], b[0]])
    plt.scatter(a[2], a[0], c="red", s= 20)
    plt.scatter(b[2], b[0], c="red", s= 20)
plt.show()


t = np.linspace(0., END_TIME, int(END_TIME*30))
t_end = END_TIME
ts = 0.1
points0 = AugmentPoints(points0)
points1 = AugmentPoints(points1)
points2 = AugmentPoints(points2)

fig, ax = plt.subplots()
plt.gca().set_aspect('equal')
plt.scatter(start_point[0], start_point[1], c="green", marker="*")
plt.scatter(end_point[0], end_point[1],     c="green", marker="*")
box_agent = patches.Rectangle((points[0,0], points[0,1]), width, width, linewidth=1.0, edgecolor='black', facecolor="none")
box_0 = patches.Rectangle((points0[0,0]-(obstacle_width-width)/2, points0[0,1]-(obstacle_width-width)/2), (obstacle_width-width), (obstacle_width-width), linewidth=1.0, edgecolor='r', facecolor="none")
box_1 = patches.Rectangle((points1[0,0]-(obstacle_width-width)/2, points1[0,1]-(obstacle_width-width)/2), (obstacle_width-width), (obstacle_width-width), linewidth=1.0, edgecolor='r', facecolor="none")
box_2 = patches.Rectangle((points2[0,0]-(obstacle_width-width)/2, points2[0,1]-(obstacle_width-width)/2), (obstacle_width-width), (obstacle_width-width), linewidth=1.0, edgecolor='r', facecolor="none")

middle = patches.Rectangle((1.5 + width/2, 1.8 + width/2), 2.5-1.5 - width/2, 2.2-1.8 - width/2, linewidth=1.0, edgecolor='black', facecolor="none")

txt = plt.text(0.9, 0.99, 't=0.',ha='right', va='top', transform=ax.transAxes)
patch_box_agent = ax.add_patch(box_agent)
patch_box_0 = ax.add_patch(box_0)
patch_box_1 = ax.add_patch(box_1)
patch_box_2 = ax.add_patch(box_2)
patch_box_3 = ax.add_patch(middle)

plt.xlim([0,4])
plt.ylim([0,4])
plt.xlabel("pos (x)")
plt.ylabel("pos (y)")
plt.title("Simulation, 3 obstacles")
fig.tight_layout()
    

def update(frame):
    time = t[frame]
    x_a, y_a = getXYfromLinePath(time, points, 2)
    x_0, y_0 = getXYfromLinePath(time, points0, 2)
    x_1, y_1 = getXYfromLinePath(time, points1, 2)
    x_2, y_2 = getXYfromLinePath(time, points2, 2)
    patch_box_agent.set_x(x_a-width/2)
    patch_box_agent.set_y(y_a-width/2)
    patch_box_0.set_x(x_0-width/2)
    patch_box_0.set_y(y_0-width/2)
    patch_box_1.set_x(x_1-width/2)
    patch_box_1.set_y(y_1-width/2)
    patch_box_2.set_x(x_2-width/2)
    patch_box_2.set_y(y_2-width/2)

    timefloat= np.round(time,2)
    txt.set_text(f"t={timefloat:.2f}")
    
    return (patch_box_agent, patch_box_0,patch_box_1,patch_box_2, txt)


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=1000/30, blit=True,repeat_delay=0.1)
plt.show()
ani.save("animated_.gif")




plt.show()



