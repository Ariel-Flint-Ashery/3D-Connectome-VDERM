# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:40:56 2022

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
        x_p, y_p, z_p = [np.arange(0,3,step = h)]*3
        xsi_init = np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T
        xsi_init= xsi_init.reshape((len(x_p), len(y_p), len(z_p), 3))
        xsi_init = np.swapaxes(xsi_init, 0, 1)
        
    return xsi_init


def update_velocity(rho, h, v, W, H):
    for i in range(W):
        if i == 0 or i == W-1:
            continue
        for j in range(H):
            if j == 0 or j == H-1:
                continue
            else:
                v[0][i][j] = -(rho[i+1][j] - rho[i-1][j])/(2*h*rho[i][j])
                v[1][i][j] = -(rho[i][j+1] - rho[i][j-1])/(2*h*rho[i][j])
    return v

def timestep(v, h):
    dt = (2*h)/(3*max(abs(v[0].flatten()) + abs(v[1].flatten())))
    return dt


def _rescale_2D_shape(polygons):

    #currently supports one filled body (body can be segmented)
    points_all = np.concatenate(([polygon for polygon in polygons]))
    xmin = min(points_all[:,0])
    xmax = max(points_all[:,0])
    ymin = min(points_all[:,1])
    ymax = max(points_all[:,1])
    print(xmin,xmax)
    print(ymin, ymax)
    Lx = abs(xmax - xmin)
    Ly = abs(ymax - ymin)
    print(Lx, Ly)
    
    #rescale shape
    alpha = [1/Lx, 1/Ly]
    for polygon in polygons:
        polygon[:,0] *= alpha[0]/3
        polygon[:,1] *= alpha[1]/3
        
        #translate shape
        polygon[:,0] += -xmin + (1/3)*Lx
        polygon[:,1] += -ymin + (1/3)*Ly
        
    return polygons

def rho_update(rho_init, W, H, h, dt):
    rho = rho_init
    rho_temp = rho.copy()
    for i in range(W):
        for j in range(H):
            if (i == 0 and j != 0 and j != H-1):
                try:
                    #print("first error", j)
                    tri = (1/h**2)*(rho[i+1][j]+rho[i][j]
                                +rho[i][j+1]+rho[i][j-1]-4*rho[i][j])
                except:
                    print("first error")
            elif (i == W-1 and j != 0 and j != H-1):
                try:
                    #print("second error")
                    tri = (1/h**2)*(rho[i][j]+rho[i-1][j]
                                +rho[i][j+1]+rho[i][j-1]-4*rho[i][j])
                except:
                    print("second error")
            
            elif (j == 0 and i != 0 and i != W-1):
                try:
                    tri = (1/h**2)*(rho[i+1][j]+rho[i-1][j]
                                +rho[i][j+1]+rho[i][j]-4*rho[i][j])
                except:
                    pass           
            elif (j == H-1 and i != 0 and i != W-1):
                try:
                    tri = (1/h**2)*(rho[i+1][j]+rho[i-1][j]+rho[i][j]
                                +rho[i][j-1]-4*rho[i][j])
                except:
                    pass

            elif (i == 0 and j == 0):
                tri = (1/h**2)*(rho[i+1][j]+rho[i][j]
                            +rho[i][j+1]+rho[i][j]-4*rho[i][j])
                
            elif (i == 0 and j == H-1):
                #print('EXPECTED END', i, j)
                tri = (1/h**2)*(rho[i+1][j]+rho[i][j]
                                +rho[i][j]+rho[i][j-1]-4*rho[i][j])
                #print(tri)
            elif (i == W-1 and j == 0):
                tri = (1/h**2)*(rho[i][j]+rho[i-1][j]
                            +rho[i][j+1]+rho[i][j]-4*rho[i][j])
                
            elif (i == W-1 and j == H-1):
                tri = (1/h**2)*(rho[i][j]+rho[i-1][j]
                            +rho[i][j]+rho[i][j-1]-4*rho[i][j])
            
            else:
                #print(i,j)
                #print("HERE ACTUALLY", i, j)
                tri = (1/h**2)*(rho[i+1][j]+rho[i-1][j]
                                +rho[i][j+1]+rho[i][j-1]-4*rho[i][j])
                
            A = 1 - dt*tri
            rho_temp[i][j] = (1/A)*rho[i][j]
    return rho_temp

#update map


def rho_update2(rho_init, W, H, h, dt):
    rho = rho_init
    rho_temp = rho.copy()
    
    for j in range(1, H-1):
        tri_bc_0 = (1/h**2)*(rho[1][j]+rho[0][j]
                    +rho[0][j+1]+rho[0][j-1]-4*rho[0][j])
        rho_temp[0][j] = rho[0][j] + dt*tri_bc_0
        
        tri_bc_1 = (1/h**2)*(rho[W-1][j]+rho[W-2][j]
                    +rho[W-1][j+1]+rho[W-1][j-1]-4*rho[W-1][j])
        rho_temp[W-1][j] = rho[W-1][j] + dt*tri_bc_1
        
    for i in range(1, W-1):
        tri_bc_0 = (1/h**2)*(rho[i+1][0]+rho[i-1][0]
                    +rho[i][1]+rho[i][0]-4*rho[i][0])
        
        rho_temp[i][0] = rho[i][0] + dt*tri_bc_0
        
        tri_bc_1 = (1/h**2)*(rho[i+1][H-1]+rho[i-1][H-1]+rho[i][H-1]
                    +rho[i][H-2]-4*rho[i][H-1])
        
        rho_temp[i][H-1] = rho[i][H-1] + dt*tri_bc_1
    
    #corners
    
    rho_temp[0][0] = rho[0][0] + dt*(1/h**2)*(rho[1][0]+rho[0][0]
                +rho[0][1]+rho[0][0]-4*rho[0][0])
    rho_temp[W-1][0] = rho[W-1][0] + dt*(1/h**2)*(rho[W-1][0]+rho[W-2][0]
                +rho[W-1][1]+rho[W-1][0]-4*rho[W-1][0])
    rho_temp[0][H-1] = rho[0][H-1] + dt*(1/h**2)*(rho[1][H-1]+rho[0][H-1]
                    +rho[0][H-1]+rho[0][H-2]-4*rho[0][H-1])
    rho_temp[W-1][H-1] = rho[W-1][H-1] + dt*(1/h**2)*(rho[W-1][H-1]+rho[W-2][H-1]
                +rho[W-1][H-1]+rho[W-1][H-2]-4*rho[W-1][H-1])
    
    
    for i in range(1, W-1):
        for j in range(1, H-1):
            tri = (1/h**2)*(rho[i+1][j]+rho[i-1][j]
                           +rho[i][j+1]+rho[i][j-1]-4*rho[i][j])
            
            rho_temp[i][j] = rho[i][j] + dt*tri
            

            
            #tri_bc_1 = ()
    return rho_temp

def simple_2Dlaplacian_stencil(graph_dict):
    """
    graph_dict is an adjaceny list for a graph.
    
    """
    keys=sorted(graph_dict.keys())
    size=len(keys)

    adjacency_matrix = np.zeros((size,size))
    degree_matrix = np.zeros((size,size))
    for a,b in [(keys.index(a), keys.index(b)) for a, row in graph_dict.items() for b in row]:
         adjacency_matrix[a][b] = 1
         
    degree_dict = {key: len(graph_dict.get(key)) for key in keys}
    
    for key in keys:
        degree_matrix[key][key] = degree_dict[key]
        
    laplacian_stencil = adjacency_matrix - degree_matrix
    
    return laplacian_stencil

def refmap_update(W, H, xsi_init, h, dt, v):
    xsi = xsi_init.copy()
    xsi_new = xsi_init.copy()
    for i in range(W):
        if (i == 0 or i == W-1):
            continue
        
        else:
            for j in range(H):
                #print(i,j)
                if (j == 0 or j == H-1):
                    #print("SKIP")
                    continue
                
                elif (i == 1 or i == W-2):
                    dxmin = (xsi[i][j] - xsi[i-1][j])/h
                    dxplus = (xsi[i+1][j] - xsi[i][j])/h
                    if (j == 1 or j == H-2):
                        dymin = (xsi[i][j] - xsi[i][j-1])/h
                        dyplus = (xsi[i][j+1] - xsi[i][j])/h
                    else:
                        dymin = (3*xsi[i][j] - 4*xsi[i][j-1] + xsi[i][j-2])/(2*h)
                        dyplus = (-xsi[i][j+2] + 4*xsi[i][j+1] - 3*xsi[i][j])/(2*h) 
                    
                elif (j == 1 or j == H-2):
                    #print(j, "point")
                    dymin = (xsi[i][j] - xsi[i][j-1])/h
                    dyplus = (xsi[i][j+1] - xsi[i][j])/h
                    if (i == 1 or i == W-2):
                        dxmin = (xsi[i][j] - xsi[i-1][j])/h
                        dxplus = (xsi[i+1][j] - xsi[i][j])/h
                    else:
                        dxmin = (3*xsi[i][j] - 4*xsi[i-1][j] + xsi[i-2][j])/(2*h)
                        dxplus = (-xsi[i+2][j] + 4*xsi[i+1][j] - 3*xsi[i][j])/(2*h)

                else: 
                    #print("ELSE")
                    #print(i,j)
                    #if i == 1:
                        #print(i)
                    dxmin = (3*xsi[i][j] - 4*xsi[i-1][j] + xsi[i-2][j])/(2*h)
                    dxplus = (-xsi[i+2][j] + 4*xsi[i+1][j] - 3*xsi[i][j])/(2*h)
                    dymin = (3*xsi[i][j] - 4*xsi[i][j-1] + xsi[i][j-2])/(2*h)
                    dyplus = (-xsi[i][j+2] + 4*xsi[i][j+1] - 3*xsi[i][j])/(2*h)
                
                #print(dymin)
                
                
                if v[0][i][j] > 0:
                    dx = v[0][i][j]*dxplus
                    
                #elif v[0][i][j] <= 0:
                elif v[0][i][j] <= 0:
                    dx = v[0][i][j]*dxmin
                    
                if v[1][i][j] > 0:
                    dy = v[1][i][j]*dyplus
                
                #else:
                elif v[1][i][j] <= 0:
                    dy = v[1][i][j]*dymin
                
                try:
                    xsi_new[i][j] = xsi[i][j] + dt*(dx + dy)
                except:
                    raise ValueError('h value is too small')
    return xsi_new


# def deformed_density_plot(xsi_new, rho_new):
#     #first, do it for a grid
#     cell_list = []
#     density = []
#     for i in range(W):
#         for j in range(H):
#             cell = [xsi_new[i][j], xsi_new[i+1][j], xsi_new[i+1][j+1], xsi_new[i, j+1]]
#             cell_list.append(cell)
#             density.append(rho_new[i][j])
#     plt.colormesh(cell_list, density)
#     plt.show()
    
# def density_plot(xsi, rho):
#     maxima = max(rho.flatten())
#     minima = min(rho.flatten())
#     norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
#     mapper = cm.ScalarMappable(norm=norm, cmap=cm.CMRmap)
#     for i in range(xsi.shape[0]):
#         if i == xsi.shape[0] - 1:
#             continue
#         else:
#             for j in range(xsi.shape[1]):
#                 if j == xsi.shape[1] - 1:
#                     continue
#                 else:
#                     cell = Polygon([xsi[i][j], xsi[i+1][j], xsi[i+1][j+1], xsi[i][j+1]])
#                     c = mapper.to_rgba(rho[i][j])
#                     plt.fill(*cell.exterior.xy, color = c, edgecolor = 'k')
    
#     plt.colorbar(mappable=mapper)
#     plt.show()
def density_plot(xsi, rho, grid_only = False):
    maxima = max(rho.flatten())
    minima = min(rho.flatten())
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.CMRmap)
    for i in range(xsi.shape[0]):
        if i == xsi.shape[0] - 1:
            continue
        else:
            for j in range(xsi.shape[1]):
                if j == xsi.shape[1] - 1:
                    continue
                else:
                    cell = Polygon([xsi[i][j], xsi[i+1][j], xsi[i+1][j+1], xsi[i][j+1]])
                    c = mapper.to_rgba(rho[i][j])
                    if grid_only == True:
                        plt.plot(*cell.exterior.xy, color = 'k')
                    else:
                        plt.fill(*cell.exterior.xy, color = c, edgecolor = 'k')
    
    if grid_only == False:
        plt.colorbar(mappable=mapper)
        
    plt.show()
            
def grid_mesh(xsi):
    cells = []
    for i in range(xsi.shape[0]):
        if i == xsi.shape[0] - 1:
            continue
        else:
            for j in range(xsi.shape[1]):
                if j == xsi.shape[1] - 1:
                    continue
                else:
                    cell = Polygon([xsi[i][j], xsi[i+1][j], xsi[i+1][j+1], xsi[i][j+1]])
                    cells.append(cell)
    return cells

# def grid_mesh_centroids(cells, vor_cells):
#     for cell in cells:
    
@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections & 1  

@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon,points[i])
    return D  

# def raster_density(polygons, xsi, densities,W, H, baseline =0):
#     points = np.concatenate(([row for row in xsi]))
#     #markers = np.zeros_like(points)
#     rhos = []
#     for polygon, density in zip(polygons, densities):
#         D = is_inside_sm_parallel(points, polygon)
#         rhos.append((np.array(D).reshape((W,H)))*density)
    
#     rho = sum(rhos) + baseline
    
#     return rho


def raster_density(polygons, xsi, densities,W, H,h, baseline =0):
    points = np.concatenate(([row for row in xsi])) + h/2
    #centroids = [[point[0] + h/2, point[1] + h/2]]
    #markers = np.zeros_like(points)
    rhos = []
    for polygon, density in zip(polygons, densities):
        D = is_inside_sm_parallel(points, polygon)
        rhos.append((np.array(D).reshape((W,H)))*density)
    
    rho = sum(rhos) + baseline
    
    return rho, D

def diffusion(rho_init, h, W, H, epsilon, xsi_init):
    v0 = (np.zeros_like(rho_init), np.zeros_like(rho_init))
    v0 = update_velocity(rho_init, h, v0, W, H)
    dt = timestep(v0, h)
    #dt = 0.25
    print(dt)
    print("INITIALISATION COMPLETE")
    rho_0 = rho_init.copy()
    rho_1 = rho_update2(rho_0, W, H, h, dt)
    print("INITIAL DENSITY UPDATE COMPLETE")
    v = update_velocity(rho_0, h, v0, W, H)
    print("INITIAL VELOCITY UPDATE COMPLETE")
    xsi_new = refmap_update(W, H, xsi_init, h, dt, v)
    print("FIRST REFERENCE MAP CREATED")
    
    #criterion = all(x>epsilon for x in np.sqrt((rho_1 - rho_0)**2).flatten()/np.mean(rho_0.flatten()))
    n=1
    print('LOOP BEGINS')
    v_mean = [[np.mean(v[0].flatten())],[np.mean(v[1].flatten())]]
    while n<1000:
        rho_0 = rho_1.copy()
        rho_1 = rho_update2(rho_0, W, H, h, dt)
        v = update_velocity(rho_0, h, v, W, H)
        v_mean[0].append(np.mean(v[0].flatten()))
        v_mean[1].append(np.mean(v[1].flatten()))
        xsi_new = refmap_update(W, H, xsi_new, h, dt, v)
        #criterion = all(x>epsilon for x in np.sqrt((rho_1 - rho_0)**2).flatten()/np.mean(rho_0.flatten()))
        n += 1
        
        # #print("""
        #       -------------------
              
        #       ITERATION NUMBER %s COMPLETED
              
              
        #       """ % (n))
        #if n == 1000:
            #break
    
    return xsi_new, rho_1, v_mean

#%%

"LEGACY TESTS"


#plot initial state
# xsi_init_points = np.concatenate(([row for row in xsi_init]))
# xsi_init_points_x = [point[0] for point in xsi_init_points]
# xsi_init_points_y = [point[1] for point in xsi_init_points]

#plt.plot(xsi_init_points_x, xsi_init_points_y, '.', color = 'k')

# xsi_init_points_x = np.zeros((xsi_init.shape[0], xsi_init.shape[1]))
# xsi_init_points_y = np.zeros((xsi_init.shape[0], xsi_init.shape[1]))
# #%%
# for i in range(xsi_init.shape[0]):
#     for j in range(xsi_init.shape[1]):
#         xsi_init_points_x[i][j] =  xsi_init[i][j][0]
#         xsi_init_points_y[i][j] = xsi_init[i][j][1]
        
# #%%
# plt.pcolormesh(xsi_init_points_x, xsi_init_points_y, rho, cmap = 'CMRmap')
# plt.colorbar()
# plt.plot(xsi_init_points_x, xsi_init_points_y, '.', color = 'k')
# plt.show()


