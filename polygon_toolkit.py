# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:34:19 2022

@author: ariel
"""

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas as gpd
import random as random
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm
import seaborn as sns
from numba import jit, njit
import numba
import matplotlib.cm as cm
import matplotlib
import open3d as o3d

#%%
bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 200)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)
    
    
def Random_Points_in_Bounds(polygon, N):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < N:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def get_polygons_np(polygons):
    polygons_np = []
    for polygon in polygons:
        polygons_np.append(np.array(polygon.exterior.coords))
    return polygons_np

def rescale_2D_shape(polygons):

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
    scale = max([Lx, Ly])
    
    #rescale shape
    #alpha = [1/Lx, 1/Ly]
    alpha = 1/scale
    for polygon in polygons:
        #translate shape
        polygon[:,0] += -xmin + scale #*(1/3)*Lx
        polygon[:,1] += -ymin + scale #*(1/3)*Ly
        
        polygon[:,0] *= alpha/3
        polygon[:,1] *= alpha/3
        
        
    return polygons

#%%
"3D shapes"

def _get_voxel_surface_cloud(file, size):
    mesh = o3d.io.read_triangle_mesh(file)
    scale = np.max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.scale(1 / scale,
               center=mesh.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                                  voxel_size=size)
    
    #bounds = [voxel_grid.get_min_bound(), voxel_grid.get_max_bound()]
    point_cloud_np = np.asarray([voxel_grid.origin + pt.grid_index*voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])
    
    for i in range(3):
        point_cloud_np[:,i] -= min(point_cloud_np[:,i])
    # alpha = 1/scale
    # #recentre point cloud
    # for i in range(3):
    #     point_cloud_np[:,i] += -1*min(point_cloud_np[:,i]) + scale
    #     point_cloud_np[:,i] *= alpha/3
        
    #     #translate shape

    #     #point_cloud_np[:,i] -= bounds[0][i]
    point_cloud_np += np.array([1,1,1])
        
    return point_cloud_np
        
        
def _fill_point_cloud(pcd, xsi):
    inside = np.zeros((xsi.shape[0],xsi.shape[1], xsi.shape[2]))
    for i in tqdm(range(xsi.shape[0])):
        for j in range(xsi.shape[1]):
            for k in range(xsi.shape[2]):
                #check surface terms
                coord = xsi[i][j][k]
                checks = 0 #checks passed successfully
                if coord in pcd:
                    inside[i][j][k] += 1
                    #continue
    
                # #check x
                # pcd_x = [point for point in pcd if (point[1] == coord[1] and point[2] == coord[2])]
                # pcd_x_r = [True for pnt in pcd_x if (pnt[0] > coord[0])]
                # pcd_x_l = [True for pnt in pcd_x if (pnt[0] < coord[0])]
                
                # if (np.mod(np.sum(pcd_x_r),2) != 0 and np.mod(np.sum(pcd_x_l),2) != 0):
                #     checks += 1
                    
                # #check y
                # pcd_y = [point for point in pcd if (point[0] == coord[0] and point[2] == coord[2])]
                # pcd_y_r = [True for pnt in pcd_y if (pnt[1] > coord[1])]
                # pcd_y_l = [True for pnt in pcd_y if (pnt[1] < coord[1])]
                
                # if (np.mod(np.sum(pcd_y_r),2) != 0 and np.mod(np.sum(pcd_y_l),2) != 0):
                #     checks += 1
                    
                # #check z
                
                # pcd_z = [point for point in pcd if (point[0] == coord[0] and point[1] == coord[1])]
                # pcd_z_r = [True for pnt in pcd_z if (pnt[2] > coord[2])]
                # pcd_z_l = [True for pnt in pcd_z if (pnt[2] < coord[2])]
                
                # if (np.mod(np.sum(pcd_z_r),2) != 0 and np.mod(np.sum(pcd_z_l),2) != 0):
                #     checks += 1
                    
                # if checks == 3:
                #     inside[i][j][k] += 1
                
                    
    return inside
    
