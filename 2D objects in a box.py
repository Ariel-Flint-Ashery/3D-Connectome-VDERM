# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:05:14 2022

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
#from diffusion_VDERM import diffusion

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
    
    
    

#%%
fig, ax = plt.subplots()
ax.set_aspect("equal")

rad = 0.5
edgy = 0.3
polygons = []
for c in np.array([[0,1]]):

    a = get_random_points(n=7, scale=1) + c
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    plt.plot(x,y)
    #polygon_temp = plt.fill(x,y)
    xy = np.vstack((x,y)).T
    polygon_temp = Polygon(xy)
    polygons.append(polygon_temp)
plt.show()



#%%

def Random_Points_in_Bounds(polygon, N):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < N:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

#%%
points = Random_Points_in_Bounds(polygon = polygons[0], N = 10)
x_points = [pnt.x for pnt in points]
y_points = [pnt.y for pnt in points]
plt.plot(*polygons[0].exterior.xy)
plt.scatter(x_points, y_points)
plt.show()

#%%


# points = np.vstack((x_points, y_points)).T
# vor = Voronoi(points)
# fig = voronoi_plot_2d(vor)
# plt.plot(*polygons[0].exterior.xy)
# plt.show()


#%%

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
#%%
points = np.vstack((x_points, y_points)).T
vor = Voronoi(points)
fig = voronoi_plot_2d(vor)
plt.plot(*polygons[0].exterior.xy)
plt.show()

#%%
# compute Voronoi tesselation
#vor = Voronoi(points)

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)
#print "--"
#print regions
#print "--"
#print vertices
#region = list of index of vertices that define it
#vertices = complete list of vertices
# colorize
# polygon_list = []
# for region in regions:
#     polygon = vertices[region]
#     plt.fill(*zip(*polygon), alpha=0.4)


 #clips tesselation to the mask
vor_cells = []
new_vertices = []
for region in regions:
    poly_reg = vertices[region]
    shape = list(poly_reg.shape)
    print(shape)
    shape[0] += 1
    p = Polygon(np.append(poly_reg, poly_reg[0]).reshape(*shape)).intersection(polygons[0])
    poly = (np.array(p.exterior.coords)).tolist()
    new_vertices.append(poly)
    vor_cells.append(p)
#%%
for poly in new_vertices:
    plt.fill(*zip(*poly), alpha=0.7)

plt.plot(points[:,0], points[:,1], 'ko')
#plt.xlim(vor.min_bound[0] - 0.5, vor.max_bound[0] + 0.5)
#plt.ylim(vor.min_bound[1] - 0.5, vor.max_bound[1] + 0.5)
#plt.plot(*polygons[0].exterior.xy)

#set density.
v = [np.array(pnts) for pnts in new_vertices] # list of vertices
vertices_full = np.vstack(v)
Lx = max(vertices_full[:,0]) - min(vertices_full[:,0])
Ly = max(vertices_full[:,1]) - min(vertices_full[:,1])
sea = Polygon([[min(vertices_full[:,0]) - .5*Lx, min(vertices_full[:,1]) - .5*Ly],
               [max(vertices_full[:,0]) + .5*Lx, min(vertices_full[:,1]) - .5*Ly],
               [max(vertices_full[:,0]) + .5*Lx, max(vertices_full[:,1]) + .5*Ly],
               [min(vertices_full[:,0]) - .5*Lx, max(vertices_full[:,1]) + .5*Ly]])

plt.plot(*sea.exterior.xy)

plt.show()

#%%

#Add density to each polygon

init_region_density = np.array([]) #same length as regions, with matching indexes for each region

init_density_map = []
#assign vertices density
for poly, rho in zip(new_vertices, init_region_density):
    poly = np.array([poly])
    density = np.ones_like(poly)*rho
    init_density_map.append(density)


#gx, gx_delta = np.linspace(0, Lx, W, retstep = True)
#gy, gy_delta = np.linspace(0, Ly, H, retstep = True)


#N = int
#kappx = 2*np.pi*np.fft.fftfreq(N, d = dx)
#kappy = 2*np.pi*np.fft.fftfreq(N, d = dy)
#%%

"finite difference method"


Lx, Ly = 100, 100
A = np.zeros((2,2))
W, H = 100, 100 #number of boxes in each dimension
gx, gx_delta = np.linspace(0, Lx, W, retstep = True)
gy, gy_delta = np.linspace(0, Ly, H, retstep = True)
h = 1 #step
rho = np.ones((W,H)) #initial density matrix


for i in range(W):
    for j in range(H):
        rho[i][j] = 10 + 9.99*np.sin((4*np.pi*i)/(W-1))*np.cos((2*np.pi*j)/(H-1))
        #rho[i][j] = 

v = (np.zeros_like(rho), np.zeros_like(rho))

#%%
def update_velocity(rho, h, v):
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


#%%

#update function
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
            
#%%

#might be necessary to do other forms of grid setups?
#for regular polygons later on we will need to create a filter to
#transform the straight line boundaries to 'pixelised' boxes. 
#maybe use Fourier transforms?
def grid_refmap_init(W,H,h):
    columns = []
    for i in tqdm(range(W)):
        column = np.zeros((H,2))
        for j in range(H):
            column[j] = np.array([i,j])*h
        #col_arr = np.array(column)
        columns.append(column)
    
    xsi_init = np.stack(columns)
    return xsi_init

#%%
#update map

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
                
                #try the else, and then do the above conditions!!
                #the problem is that we only call one of the conditions
                #i.e. if above gets called, then dxmin or dxplus are undefined. 
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
                    dx = v[0][i][j]*dxmin
                    
                #elif v[0][i][j] <= 0:
                else:
                    dx = v[0][i][j]*dxplus
                    
                if v[1][i][j] > 0:
                    dy = v[1][i][j]*dymin
                
                else:
                #elif v[1][i][j] <= 0:
                    dy = v[1][i][j]*dyplus
                    
                xsi_new[i][j] = xsi[i][j] + dt*(dx + dy)
    return xsi_new
        
                #FIX BROKEN LOGIC
#%%

#convergence criteria
#epsilon = 0.1

def diffusion(rho_init, h, W, H, epsilon, xsi_init):
    v0 = (np.zeros_like(rho_init), np.zeros_like(rho_init))
    v0 = update_velocity(rho_init, h, v0)
    dt = timestep(v0, h)
    
    print("INITIALISATION COMPLETE")
    rho_0 = rho_init.copy()
    rho_1 = rho_update(rho_0, W, H, h, dt)
    print("INITIAL DENSITY UPDATE COMPLETE")
    v = update_velocity(rho_0, h, v0)
    print("INITIAL VELOCITY UPDATE COMPLETE")
    xsi_new = refmap_update(W, H, xsi_init, h, dt, v)
    print("FIRST REFERENCE MAP CREATED")
    
    #criterion = all(x>epsilon for x in np.sqrt((rho_1 - rho_0)**2).flatten()/np.mean(rho_0.flatten()))
    n=1
    print('LOOP BEGINS')
    while tqdm(n<1000):
        rho_0 = rho_1
        rho_1 = rho_update(rho_0, W, H, h, dt)
        v = update_velocity(rho_0, h, v)
        xsi_new = refmap_update(W, H, xsi_new, h, dt, v)
        #criterion = all(x>epsilon for x in np.sqrt((rho_1 - rho_0)**2).flatten()/np.mean(rho_0.flatten()))
        n += 1
        #if n == 1000:
            #break
    return xsi_new, rho_1

#%%
#TEST
#Lx, Ly = 100, 100
#A = np.zeros((2,2))
W, H = 10, 10 #number of boxes in each dimension
##gx, gx_delta = np.linspace(0, Lx, W, retstep = True)
#gy, gy_delta = np.linspace(0, Ly, H, retstep = True)
h = 1 #step
rho = np.ones((W,H)) #initial density matrix


for i in range(W):
    for j in range(H):
        rho[i][j] = 10 + 9.99*np.sin((4*np.pi*i*h)/(W-1))*np.cos((2*np.pi*j*h)/(H-1))
        #rho[i][j] = 

v = (np.zeros_like(rho), np.zeros_like(rho))
xsi_init = grid_refmap_init(W, H, h)
xsi_new, rho_new = diffusion(rho, h, W, H, 0.5, xsi_init)


#%%
xsi_rows = [row for row in xsi_new]
print(xsi_rows[0])
#%%
# xsi_points = np.concatenate(([row for row in xsi_new]))
# xsi_points_x = [point[0] for point in xsi_points]
# xsi_points_y = [point[1] for point in xsi_points]
# plt.plot(xsi_points_x, xsi_points_y, '.', color = 'r')

xsi_init_points = np.concatenate(([row for row in xsi_init]))
xsi_init_points_x = [point[0] for point in xsi_init_points]
xsi_init_points_y = [point[1] for point in xsi_init_points]

plt.plot(xsi_init_points_x, xsi_init_points_y, '.', color = 'k')
sns.heatmap(rho)
#sns.heatmap(rho_new)
plt.show()
#%%
xsi_points = np.concatenate(([row for row in xsi_new]))
xsi_points_x = [point[0] for point in xsi_points]
xsi_points_y = [point[1] for point in xsi_points]
plt.plot(xsi_points_x, xsi_points_y, '.', color = 'k')

#sns.heatmap(rho_new)
#sns.heatmap(rho_new)
plt.show()

#%%
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

def grid_refmap_init(W,H,h):
    columns = []
    for i in tqdm(range(W)):
        column = np.zeros((H,2))
        for j in range(H):
            column[j] = np.array([i,j])*h
        #col_arr = np.array(column)
        columns.append(column)
    
    xsi_init = np.stack(columns)
    return xsi_init
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
#%%
N = 1000
points = np.random.uniform(-1, 3, size=(N, 2))
x_points = [point[0] for point in points]
y_points = [point[1] for point in points]

poly = np.array(polygons[0].exterior.coords)
inside = is_inside_sm_parallel(points, poly)
#%%

plt.scatter(x_points, y_points, c=inside, ec = 'k')
plt.show()

#%%
W, H = 30, 30 #number of boxes in each dimension
h = 0.1 #step
#create grid
xsi_init = grid_refmap_init(W, H, h)

#initialise density
rho_init, D = raster_density([poly], xsi_init, [3], W, H, h, baseline = 1)

#%%
density_plot(xsi_init, rho_init)

#%%

xsi_new, rho_new, v_mean_final = diffusion(rho_init, h, W, H, 0.5, xsi_init)

#%%

density_plot(xsi_new, rho_new, grid_only = True)

#%%
rescaled_poly = _rescale_2D_shape([poly])

#%%
for poly in rescaled_poly:
    p = Polygon(poly)
    plt.plot(*p.exterior.xy)
    plt.show()
    