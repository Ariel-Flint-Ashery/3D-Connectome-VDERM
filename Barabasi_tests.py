# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:30:21 2022

@author: ariel
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from shapely.geometry import Point, Polygon
import os
import matplotlib
import matplotlib.cm as cm
from numba import jit, njit
import numba
from time import time
import diffusion_VDERM as brb
import polygon_toolkit as pt
import open3d as o3d
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
#%%
#RUN TESTS
W, H = 20, 20 #number of boxes in each dimension
h = 0.5 #step
rho = np.ones((W,H)) #create initial density array

#set up density 
for i in range(W):
    for j in range(H):
        rho[i][j] = 10 + 9.99*np.sin((4*np.pi*i*h)/(W-1))*np.cos((2*np.pi*j*h)/(H-1))
        #rho[i][j] = 
#create initial velocity array
v = (np.zeros_like(rho), np.zeros_like(rho))
#create initial grid
xsi_init = brb.grid_refmap_init([W, H], h, d=2)
#diffuse
xsi_new, rho_new, v_mean_final = brb.diffusion(rho, h, W, H, 0.5, xsi_init)

#%%
#test density map plot function
brb.density_plot(xsi_init, rho)

brb.density_plot(xsi_new, rho_new)
#%%
#find initial and final mean density

rho_init_mean = np.mean(rho.flatten())
rho_final_mean = np.mean(rho_new.flatten())

print(rho_init_mean)
print(rho_final_mean)

#%%

#TEST INSIDE_SM_PARALLEL CODE
np.random.seed(2)
time_is_inside_sm_parallel=[]
n_points=[]

for i in range(1, 10000002, 1000000): 
    n_points.append(i)
    
    lenpoly = 100
    polygon = [[np.sin(x)+0.5,np.cos(x)+0.5] for x in np.linspace(0,2*np.pi,lenpoly)]
    polygon = np.array(polygon)
    N = i
    points = np.random.uniform(-1.5, 1.5, size=(N, 2))
    
    start_time = time()
    inside4=brb.is_inside_sm_parallel(points,polygon)
    time_is_inside_sm_parallel.append(time()-start_time)

plt.plot(n_points,time_is_inside_sm_parallel,label='is_inside_sm_parallel')

plt.xlabel("N points")
plt.ylabel("time (sec)")
plt.legend(loc = 'best')
plt.show()

#%%
"TEST FOR 2D POLYGONS"

#%%
#create polygon

fig, ax = plt.subplots()
ax.set_aspect("equal")
rad = 0.5
edgy = 0.3
polygons = []
for c in np.array([[0,0]]):
    a = pt.get_random_points(n=7, scale=1) + c
    x,y, _ = pt.get_bezier_curve(a,rad=rad, edgy=edgy)
    plt.plot(x,y)
    #polygon_temp = plt.fill(x,y)
    xy = np.vstack((x,y)).T
    polygon_temp = Polygon(xy)
    polygons.append(polygon_temp)
plt.show()

#%%
# check is_inside_sm_parallel function

# N = 1000
# points = np.random.uniform(-1, 3, size=(N, 2))
# x_points = [point[0] for point in points]
# y_points = [point[1] for point in points]


# poly = np.array(polygons[0].exterior.coords)

# inside = brb.is_inside_sm_parallel(points, poly)

# plt.scatter(x_points, y_points, c=inside, ec = 'k')
# plt.show()

#%%

poly = pt.get_polygons_np(polygons)

rescaled_poly = pt.rescale_2D_shape(poly)
#%%
for poly in rescaled_poly:
    p = Polygon(poly)
    plt.plot(*p.exterior.xy)
    plt.show()
    

#%%
h = 0.05 #step
W, H = [int(1/h)]*2 #number of boxes in each dimension
#create grid
xsi_init = brb.grid_refmap_init([W, H], h, d = 2)
#%%
#initialise density
rho_init, D = brb.raster_density(rescaled_poly, xsi_init, [3], W, H, h, baseline = 1)

brb.density_plot(xsi_init, rho_init, grid_only=False)

#%%
xsi_new, rho_new, v_mean_final = brb.diffusion(rho_init, h, W, H, 0.5, xsi_init)

#%%
brb.density_plot(xsi_new, rho_new, grid_only = True)

#%%
"3D TESTS"
#load surface point cloud
pcd = pt._get_voxel_surface_cloud('Stanford_Bunny.stl', 0.05)

#%%
#PLOT PCD
x_points = pcd[:,0]
y_points = pcd[:,1]
z_points = pcd[:,2]

# x_points -= min(x_points)
# z_points -= min(z_points)
# y_points -= min(y_points)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x_points, y_points, z_points)
plt.show()

#%%
h = 0.05 #step
W, H, L = [int(1/h)]*3 #number of boxes in each dimension
#create grid
xsi_init = brb.grid_refmap_init([W, H, L], h, d = 3)

#%%

pcd_dense = pt._fill_point_cloud(pcd, xsi_init)

#%%

rho_init = pcd_dense*2 + 1
#%%

xsi_init_points = np.concatenate([item for sublist in xsi_init for item in sublist])
xsi_init_points_x = [point[0] for point in xsi_init_points]
xsi_init_points_y = [point[1] for point in xsi_init_points]
xsi_init_points_z = [point[2] for point in xsi_init_points]

#%%
x_points = xsi_init_points[:,0]
y_points = xsi_init_points[:,1]
z_points = xsi_init_points[:,2]
#%%
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x_points, y_points, z_points, c = rho_init.flatten())
plt.show()

#%%
pcd_rounded = np.around(pcd, decimals =2)
rho = np.zeros((xsi_init.shape[0],xsi_init.shape[0],xsi_init.shape[0]))
count = 0
for i in range(xsi_init.shape[0]):
    for j in range(xsi_init.shape[0]):
        for k in range(xsi_init.shape[0]):
            if np.around(xsi_init[i][j][k], decimals =2) in pcd_rounded:
                count += 1
                rho[i][j][k] += 1
                
#%%
rho = np.zeros((xsi_init.shape[0],xsi_init.shape[0],xsi_init.shape[0]))
for point in pcd:
    if point in xsi_init:
        indices = point/0.05
        rho[int(indices[0])][int(indices[1])][int(indices[2])] += 1
#%%
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x_points, y_points, z_points, c = rho.flatten())
plt.show()
#%%
xb_points = pcd[:,0]
yb_points = pcd[:,1]
zb_points = pcd[:,2]
#%%
# x_points -= min(x_points)
# z_points -= min(z_points)
# y_points -= min(y_points)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x_points, y_points, z_points, c= 'k', marker = '.', alpha = 0.05)
ax.scatter(xb_points, yb_points, zb_points, c = 'r', marker = '.')

#fig = plt.figure(figsize=(12, 12))
#ax = fig.add_subplot(projection='3d')
#ax.scatter(x_points, y_points, z_points, 'k.')
#ax.scatter
plt.show()
#%%
plt.scatter(x_points, z_points, c= 'k', marker = '.', alpha = 0.05)
plt.scatter(xb_points, zb_points, c = 'r', marker = '.')
plt.show()

#%%

