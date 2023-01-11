# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:28:53 2022

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
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#%%

def grid_refmap_init(dim, h, d):
    if d == 2:
        columns = []
        for i in tqdm(range(dim[0])):
            column = np.zeros((dim[1],2))
            for j in range(dim[1]):
                column[j] = np.array([i,j])*h
            #col_arr = np.array(column)
            columns.append(column)
        
        xsi_init = np.stack(columns)
        
    if d == 3:
        x_p, y_p, z_p = [np.arange(0,x, step = h).astype('float') for x in dim]#[np.arange(0,3,step = h)]*3
        xsi_init = np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T
        xsi_init= xsi_init.reshape((len(x_p), len(y_p), len(z_p), 3))
        xsi_init = np.swapaxes(xsi_init, 0, 1)
        
    return xsi_init

def update_velocity(rho, h, v, dim):
    for j in range(dim[1]):
        for k in range(dim[2]):
            for i in range(dim[0]):
                if i != 0 and i != (dim[0]-1):
                    v[0][i][j][k] = -1*(rho[i+1][j][k] - rho[i-1][j][k])/(2*h*rho[i][j][k])
    for i in range(dim[0]):
        for k in range(dim[2]):
            for j in range(dim[1]):
                if j != 0 and j != (dim[1] - 1):
                    v[1][i][j][k] = -1*(rho[i][j+1][k] - rho[i][j-1][k])/(2*h*rho[i][j][k])
                    
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                if k != 0 and k != (dim[2]-1):
                    v[2][i][j][k] = -1*(rho[i][j][k+1] - rho[i][j][k-1])/(2*h*rho[i][j][k])
                
    # for i in range(dim[0]):
    #     xbound = False
    #     if i == 0 or i == (dim[0] - 1):
    #         xbound = True
    #     for j in range(dim[1]):
    #         ybound = False
    #         if j == 0 or j == (dim[1] - 1):
    #             ybound = True
    #         for k in range(dim[2]):
    #             zbound = False
    #             if k == 0 or k == (dim[2] - 1):
    #                 zbound = True
                    
    #             if xbound == False:
    #                 v[0][i][j][k] = -1*(rho[i+1][j][k] - rho[i-1][j][k])/(2*h*rho[i][j][k])
    #             if ybound == False:
    #                 v[1][i][j][k] = -1*(rho[i][j+1][k] - rho[i][j-1][k])/(2*h*rho[i][j][k])
    #             if zbound == False:
    #                 v[2][i][j][k] = -1*(rho[i][j][k+1] - rho[i][j][k-1])/(2*h*rho[i][j][k])
    
    return v



def timestep(v, h):
    dt = (2*h)/(3*max(abs(v[0].flatten()) + abs(v[1].flatten()) + abs(v[2].flatten())))
    return dt


def rhoUpdate3D(rho_init,dim, h, dt):
    rho = rho_init
    rho_temp = rho.copy()
    W, H, L = dim
    #i bounds
    for j in range(1, H-1):
        for k in range(1, L-1):
            tri_bc_0 = (1/h**2)*(rho[1][j][k]+rho[0][j][k]
                        +rho[0][j+1][k]+rho[0][j-1][k]
                        +rho[0][j][k+1]+rho[0][j][k-1]
                        -6*rho[0][j][k])
            rho_temp[0][j][k] = rho[0][j][k] + dt*tri_bc_0
            
            tri_bc_1 = (1/h**2)*(rho[W-1][j][k]+rho[W-2][j][k]
                        +rho[W-1][j+1][k]+rho[W-1][j-1][k]
                        +rho[W-1][j][k-1]+rho[W-1][j][k+1]
                        -6*rho[W-1][j][k])
            rho_temp[W-1][j][k] = rho[W-1][j][k] + dt*tri_bc_1
    #j bounds
    for i in range(1, W-1):
        for k in range(1, L-1):
            tri_bc_0 = (1/h**2)*(rho[i+1][0][k]+rho[i-1][0][k]
                        +rho[i][1][k]+rho[i][0][k]
                        +rho[i][0][k-1]+rho[i][0][k+1]
                        -6*rho[i][0][k])
            
            rho_temp[i][0][k] = rho[i][0][k] + dt*tri_bc_0
            
            tri_bc_1 = (1/h**2)*(rho[i+1][H-1][k]+rho[i-1][H-1][k]
                        +rho[i][H-1][k]+rho[i][H-2][k]
                        +rho[i][H-1][k-1]+rho[i][H-1][k+1]
                        -6*rho[i][H-1][k])
            
            rho_temp[i][H-1][k] = rho[i][H-1][k] + dt*tri_bc_1
    
    #k bounds     
    for i in range(1, W-1):
        for j in range(1, H-1):
            tri_bc_0 = (1/h**2)*(rho[i+1][j][0]+rho[i-1][j][0]
                                 +rho[i][j+1][0]+rho[i][j-1][0]
                                 +rho[i][j][1]+rho[i][j][0]
                                 -6*rho[i][j][0])
            
            rho_temp[i][j][0] = rho[i][j][0] + dt*tri_bc_0
            
            tri_bc_1 = (1/h**2)*(rho[i+1][j][L-1]+rho[i-1][j][L-1]
                                 +rho[i][j+1][L-1]+rho[i][j-1][L-1]
                                 +rho[i][j][L-1]+rho[i][j][L-2]
                                 -6*rho[i][j][L-1])
            
            rho_temp[i][j][L-1] = rho[i][j][L-1] + dt*tri_bc_1
    
    #corners
    
    rho_temp[0][0][0] = rho[0][0][0] + dt*(1/h**2)*(rho[1][0][0]+rho[0][0][0]
                +rho[0][1][0]+rho[0][0][0]+rho[0][0][1]+rho[0][0][0]-6*rho[0][0][0])
    rho_temp[W-1][0][0] = rho[W-1][0][0] + dt*(1/h**2)*(rho[W-1][0][0]+rho[W-2][0][0]
                +rho[W-1][1][0]+rho[W-1][0][0]+rho[W-1][0][1]+rho[W-1][0][0]-6*rho[W-1][0][0])
    rho_temp[0][H-1][0] = rho[0][H-1][0] + dt*(1/h**2)*(rho[1][H-1][0]+rho[0][H-1][0]
                    +rho[0][H-1][0]+rho[0][H-2][0]+rho[0][H-1][1]+rho[0][H-1][0]-6*rho[0][H-1][0])
    rho_temp[0][0][L-1] = rho[0][0][L-1] + dt*(1/h**2)*(rho[1][0][L-1]+rho[0][0][L-1]
                            +rho[0][1][L-1]+rho[0][0][L-1]+rho[0][0][L-1]+rho[0][0][L-2]-6*rho[0][0][L-1])
    rho_temp[W-1][H-1][L-1] = rho[W-1][H-1][L-1] + dt*(1/h**2)*(rho[W-1][H-1][L-1]+rho[W-2][H-1][L-1]
                            +rho[W-1][H-1][L-1]+rho[W-1][H-2][L-1]+rho[W-1][H-1][L-1]+rho[W-1][H-1][L-2]-6*rho[W-1][H-1][L-1])
    
    
    for i in range(1, W-1):
        for j in range(1, H-1):
            for k in range(1, L-1):
                tri = (1/h**2)*(rho[i+1][j][k]+rho[i-1][j][k]+rho[i][j+1][k]
                                +rho[i][j-1][k]+rho[i][j][k+1]+rho[i][j][k-1]
                                -6*rho[i][j][k])
                
                rho_temp[i][j][k] = rho[i][j][k] + dt*tri
            

            
            #tri_bc_1 = ()
    return rho_temp
#%%

# def refmapUpdate3D(dim, xsi_init, h, dt, v):
#     xsi = xsi_init.copy()
#     xsi_new = xsi_init.copy()
#     W, H, L = dim
#     for i in range(W):
#         if (i == 0 or i == W-1):
#             continue
        
#         else:
#             for j in range(H):
#                 #print(i,j)
#                 if (j == 0 or j == H-1):
#                     #print("SKIP")
#                     continue
#                 else:
#                     for k in range(L):
#                         if (k== 0 or k == L-1):
#                             continue
    
#                         elif (i == 1 or i == W-2):
#                             dxmin = (xsi[i][j][k] - xsi[i-1][j][k])/h
#                             dxplus = (xsi[i+1][j][k] - xsi[i][j][k])/h
#                             if (j == 1 or j == H-2):
#                                 dymin = (xsi[i][j][k] - xsi[i][j-1][k])/h
#                                 dyplus = (xsi[i][j+1][k] - xsi[i][j][k])/h
#                             else:
#                                 dymin = (3*xsi[i][j][k] - 4*xsi[i][j-1][k] + xsi[i][j-2][k])/(2*h)
#                                 dyplus = (-xsi[i][j+2][k] + 4*xsi[i][j+1][k] - 3*xsi[i][j][k])/(2*h)
                                
#                             if (k == 1 or k == L-2):
#                                 dzmin = (xsi[i][j][k] - xsi[i][j][k-1])/h
#                                 dzplus = (xsi[i][j][k+1] - xsi[i][j][k])/h
                            
#                             else:
#                                 dzmin = (3*xsi[i][j][k] - 4*xsi[i][j][k-1] + xsi[i][j][k-2])/(2*h)
#                                 dzplus = (-xsi[i][j][k+2] + 4*xsi[i][j][k+1] - 3*xsi[i][j][k])/(2*h)
                                
                            
#                         elif (j == 1 or j == H-2):
#                             #print(j, "point")
#                             dymin = (xsi[i][j][k] - xsi[i][j-1][k])/h
#                             dyplus = (xsi[i][j+1][k] - xsi[i][j][k])/h
#                             if (i == 1 or i == W-2):
#                                 dxmin = (xsi[i][j][k] - xsi[i-1][j][k])/h
#                                 dxplus = (xsi[i+1][j][k] - xsi[i][j][k])/h
#                             else:
#                                 dxmin = (3*xsi[i][j][k] - 4*xsi[i-1][j][k] + xsi[i-2][j][k])/(2*h)
#                                 dxplus = (-xsi[i+2][j][k] + 4*xsi[i+1][j][k] - 3*xsi[i][j][k])/(2*h)
                            
#                             if (k == 1 or k == L-2):
#                                 dzmin = (xsi[i][j][k] - xsi[i][j][k-1])/h
#                                 dzplus = (xsi[i][j][k+1] - xsi[i][j][k])/h
                            
#                             else:
#                                 dzmin = (3*xsi[i][j][k] - 4*xsi[i][j][k-1] + xsi[i][j][k-2])/(2*h)
#                                 dzplus = (-xsi[i][j][k+2] + 4*xsi[i][j][k+1] - 3*xsi[i][j][k])/(2*h)
                        
#                         elif (k == 1 or k == L-2):
#                             dzmin = (xsi[i][j][k] - xsi[i][j][k-1])/h
#                             dzplus = (xsi[i][j][k+1] - xsi[i][j][k])/h
#                             if (i == 1 or i == W-2):
#                                 dxmin = (xsi[i][j][k] - xsi[i-1][j][k])/h
#                                 dxplus = (xsi[i+1][j][k] - xsi[i][j][k])/h
#                             else:
#                                 dxmin = (3*xsi[i][j][k] - 4*xsi[i-1][j][k] + xsi[i-2][j][k])/(2*h)
#                                 dxplus = (-xsi[i+2][j][k] + 4*xsi[i+1][j][k] - 3*xsi[i][j][k])/(2*h)
#                             if (j == 1 or j == H-2):
#                                 dymin = (xsi[i][j][k] - xsi[i][j-1][k])/h
#                                 dyplus = (xsi[i][j+1][k] - xsi[i][j][k])/h
#                             else:
#                                 dymin = (3*xsi[i][j][k] - 4*xsi[i][j-1][k] + xsi[i][j-2][k])/(2*h)
#                                 dyplus = (-xsi[i][j+2][k] + 4*xsi[i][j+1][k] - 3*xsi[i][j][k])/(2*h)
                            
#                         else: 
#                             dxmin = (3*xsi[i][j][k] - 4*xsi[i-1][j][k] + xsi[i-2][j][k])/(2*h)
#                             dxplus = (-xsi[i+2][j][k] + 4*xsi[i+1][j][k] - 3*xsi[i][j][k])/(2*h)
#                             dymin = (3*xsi[i][j][k] - 4*xsi[i][j-1][k] + xsi[i][j-2][k])/(2*h)
#                             dyplus = (-xsi[i][j+2][k] + 4*xsi[i][j+1][k] - 3*xsi[i][j][k])/(2*h)
#                             dzmin = (3*xsi[i][j][k] - 4*xsi[i][j][k-1] + xsi[i][j][k-2])/(2*h)
#                             dzplus = (-xsi[i][j][k+2] + 4*xsi[i][j][k+1] - 3*xsi[i][j][k])/(2*h)
                        
#                         #print(dymin)
                        
                        
#                         if v[0][i][j][k] > 0:
#                             dx = v[0][i][j][k]*dxplus
                            
#                         elif v[0][i][j][k] <= 0:
#                             dx = v[0][i][j][k]*dxmin
                            
#                         if v[1][i][j][k] > 0:
#                             dy = v[1][i][j][k]*dyplus
                        
#                         elif v[1][i][j][k] <= 0:
#                             dy = v[1][i][j][k]*dymin
                        
#                         if v[2][i][j][k] > 0:
#                             dz = v[2][i][j][k]*dzplus
                        
#                         elif v[2][i][j][k] <= 0:
#                             dz = v[2][i][j][k]*dzmin
                        
#                         try:
#                             xsi_new[i][j][k] = xsi[i][j][k] + dt*(dx + dy + dz)
#                         except:
#                             raise ValueError('h value is too small')
#     return xsi_new
#%%
def refmapUpdate3D(dim, xsi_init, h, dt, v):
    xsi = xsi_init.copy()
    xsi_new = xsi_init.copy()
    W, H, L = dim
    
    #corners
    for j in range(dim[1]):
        for k in range(dim[2]):
            for i in range(dim[0]):
                if i == 0 or i == (dim[0]-1):
                    dxmin, dxplus = 0.0, 0.0
                elif (i == 1 or i == (dim[0] - 2)):
                    dxmin = (xsi[i][j][k] - xsi[i-1][j][k])/h
                    dxplus = (xsi[i+1][j][k] - xsi[i][j][k])/h
                
                else:
                    dxmin = (3*xsi[i][j][k] - 4*xsi[i-1][j][k] + xsi[i-2][j][k])/(2*h)
                    dxplus = (-xsi[i+2][j][k] + 4*xsi[i+1][j][k] - 3*xsi[i][j][k])/(2*h) 
    
                if v[0][i][j][k] > 0:
                    delta_x = v[0][i][j][k]*dxplus
                    
                elif v[0][i][j][k] <= 0:
                    delta_x = v[0][i][j][k]*dxmin
                
                xsi_new[i][j][k] = xsi[i][j][k] + dt*delta_x
                
    for i in range(dim[0]):
        for k in range(dim[2]):
            for j in range(dim[1]):
                if j == 0 or j == (dim[1] - 1):
                    dymin,dyplus = 0.0,0.0
                elif j == 1 or j == (dim[1] - 2):
                    dymin = (xsi[i][j][k] - xsi[i][j-1][k])/h
                    dyplus = (xsi[i][j+1][k] - xsi[i][j][k])/h
                
                else:
                    dymin = (3*xsi[i][j][k] - 4*xsi[i][j-1][k] + xsi[i][j-2][k])/(2*h)
                    dyplus = (-xsi[i][j+2][k] + 4*xsi[i][j+1][k] - 3*xsi[i][j][k])/(2*h)
    
                if v[1][i][j][k] > 0:
                    delta_y = v[1][i][j][k]*dyplus
                    
                elif v[1][i][j][k] <= 0:
                    delta_y = v[1][i][j][k]*dymin
                
                xsi_new[i][j][k] = xsi[i][j][k] + dt*delta_y
                
                
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                if k == 0 or k == (dim[2] - 1):
                    dzmin, dzplus = 0.0, 0.0
                elif k == 1 or k == (dim[2] - 2):
                    dzmin = (xsi[i][j][k] - xsi[i][j][k-1])/h
                    dzplus = (xsi[i][j][k+1] - xsi[i][j][k])/h
                
                else:
                    dzmin = (3*xsi[i][j][k] - 4*xsi[i][j][k-1] + xsi[i][j][k-2])/(2*h)
                    dzplus = (-xsi[i][j][k+2] + 4*xsi[i][j][k+1] - 3*xsi[i][j][k])/(2*h)
                    
                if v[2][i][j][k] > 0:
                    delta_z = v[2][i][j][k]*dzplus
                    
                elif v[2][i][j][k] <= 0:
                    delta_z = v[2][i][j][k]*dzmin
                
                xsi_new[i][j][k] = xsi[i][j][k] + dt*delta_z
                
  
    return xsi_new
                    
#%%
def diffusion3D(rho_init, h, dim, iterations, xsi_init):
    v0 = (np.zeros_like(rho_init), np.zeros_like(rho_init), np.zeros_like(rho_init))
    v0 = update_velocity(rho_init, h, v0, dim)
    dt = timestep(v0, h)
    #dt = 0.25
    print(dt)
    print("INITIALISATION COMPLETE")
    rho_0 = rho_init.copy()
    rho_1 = rhoUpdate3D(rho_0, dim, h, dt)
    print("INITIAL DENSITY UPDATE COMPLETE")
    v = update_velocity(rho_0, h, v0, dim)
    print("INITIAL VELOCITY UPDATE COMPLETE")
    xsi_new = refmapUpdate3D(dim, xsi_init, h, dt, v)
    print("FIRST REFERENCE MAP CREATED")
    
    #criterion = all(x>epsilon for x in np.sqrt((rho_1 - rho_0)**2).flatten()/np.mean(rho_0.flatten()))
    n=0
    print('LOOP BEGINS')
    #v_mean = [[np.mean(v[0].flatten())],[np.mean(v[1].flatten())]]
    while n<iterations:
        rho_0 = rho_1.copy()
        rho_1 = rhoUpdate3D(rho_0, dim, h, dt)
        v = update_velocity(rho_0, h, v, dim)
        #v_mean[0].append(np.mean(v[0].flatten()))
        #v_mean[1].append(np.mean(v[1].flatten()))
        xsi_new = refmapUpdate3D(dim, xsi_new, h, dt, v)
        #criterion = all(x>epsilon for x in np.sqrt((rho_1 - rho_0)**2).flatten()/np.mean(rho_0.flatten()))
        n += 1
            
    return xsi_new, rho_1#, v_mean

#%%
#RUN TESTS
dim = [10,10,10] #number of boxes in each dimension

h = 1 #step
rho = np.ones(dim) #create initial density array

#set up density 
for i in range(dim[0]):
    for j in range(dim[1]):
        for k in range(dim[2]):
            rho[i][j][k] = 10 + 9.99*np.sin((4*np.pi*i*h)/(dim[0]-1))*np.cos((2*np.pi*j*h)/(dim[1]-1))*np.cos((2*np.pi*k*h)/(dim[2]-1))

#%%

#create initial grid
xsi_init = grid_refmap_init(dim, h, 3)
#%%
#diffuse
xsi_new, rho_new = diffusion3D(rho, h, dim, 100, xsi_init)
#%%
xsi_flat = xsi_new.flatten()
print(max(xsi_flat))
print(min(xsi_flat))

#%%
rho_init_mean = np.mean(rho.flatten())
rho_final_mean = np.mean(rho_new.flatten())

print(rho_init_mean)
print(rho_final_mean)

#%%
s1 = xsi_new[:][1]
s2 = xsi_new[:][2]
s3 = xsi_new[:][3]#s4 = xsi_new[:][6]
#%%
import mpl_toolkits.mplot3d as a3
def density_plot(data, rho, grid_only = False):
    #maxima = max(rho.flatten())
    #minima = min(rho.flatten())
    #norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    #mapper = cm.ScalarMappable(norm=norm, cmap=cm.CMRmap)
    
    ax = plt.figure().add_subplot(projection='3d')
    #ax = a3.Axes3D(plt.figure())
    for xsi in data:
        
        for i in range(xsi.shape[0]):
            if i == xsi.shape[0] - 1:
                continue
            else:
                for j in range(xsi.shape[1]):
                    if j == xsi.shape[1]-1:
                        continue
                    else:
                        cellx = [xsi[i][j][0], xsi[i+1][j][0], xsi[i+1][j+1][0], xsi[i][j+1][0]]
                        celly = [xsi[i][j][1], xsi[i+1][j][1], xsi[i+1][j+1][1], xsi[i][j+1][1]]
                        cellz = [xsi[i][j][2], xsi[i+1][j][2], xsi[i+1][j+1][2], xsi[i][j+1][2]]
                        ax.plot(cellx, celly, cellz, color = 'k')
                        ax.set_xlim(0.9, 1.1)
                        ax.set_ylim(0, 9)
                        ax.set_zlim(0, 9)
                        #tri = a3.art3d.Poly3DCollection(cell)
                        #tri.set_color(colors.rgb2hex(np.random.rand(3)))
                        #tri.set_edgecolor('k')
                        #ax.add_collection3d(cell)
    
    #plt.show()
    
    
    # for i in range(xsi.shape[0]):
    #     if i == xsi.shape[0] - 1:
    #         continue
    #     else:
    #         for j in range(xsi.shape[1]):
    #             if j == xsi.shape[1] - 1:
    #                 continue
    #             else:
    #                 cell = Polygon([xsi[i][j], xsi[i+1][j], xsi[i+1][j+1], xsi[i][j+1]])
    #                 c = mapper.to_rgba(rho[i][j])
    #                 if grid_only == True:
    #                     plt.plot(*cell.exterior.xy, color = 'k')
    #                 else:
    #                     plt.fill(*cell.exterior.xy, color = c, edgecolor = 'k')
    
    # if grid_only == False:
    #     plt.colorbar(mappable=mapper)

    
#%%

density_plot([s1, s2,s3], rho_new[2], grid_only = True)
plt.show()

#%%
density_plot([s1], rho_new[2])
#%%
x, y, z = [], [], []
for i in range(s1.shape[0]):
    for j in range(s1.shape[1]):
        x_values = s1[i][j][0]
        y_values = s1[i][j][1]
        z_values = s1[i][j][2]
        x.append(x_values)
        y.append(y_values)
        z.append(z_values)
        
#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z)
ax.set_x