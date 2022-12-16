# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:21:28 2022

@author: ariel
"""
import vtk as vtk
import numpy as np
import pyvista as pv
from pyvista import examples
import trimesh
from trimesh.voxel import creation
import os
import open3d as o3d
from pyntcloud import PyntCloud
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import matplotlib.pyplot as plt
#%%
# Load a surface to voxelize
surface = examples.download_foot_bones()
surface

voxels = pv.voxelize(surface, density=surface.length / 500)
voxels.plot(opacity=1.00)

#%%

tetra = pv.voxelize(pv.Tetrahedron(), density=0.1)
tetra.plot(scalars='vtkOriginalCellIds')

#%%
tetra_cells = tetra.GetCells()
tetra_cells_loc = tetra.GetCellLocationsArray

#%%
res = [10,10]
h = [2, 2]
x, y= np.indices((res[0], res[1]))
total_voxels = np.product(res)
coords = np.concatenate((np.reshape(x*h[0], [total_voxels, 1]),       
         np.reshape(y*h[1], [total_voxels, 1])), axis=1)

#%%
mesh = trimesh.load_mesh('Stanford_Bunny.stl')

#%%
vox_grid = creation.local_voxelize(mesh, (0,0,0), 1, 200, fill=True)

#%%
vox_mesh = vox_grid.as_boxes()
#%%
s = trimesh.Scene()
s.add_geometry(vox_mesh)
s.show()
#%%
mesh.show()
#%%

mesh = o3d.io.read_triangle_mesh('Stanford_Bunny.stl')
#%%
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
           center=mesh.get_center())
o3d.visualization.draw_geometries([mesh])

#%%
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.05)
#o3d.visualization.draw_geometries([voxel_grid])

#%%


#%%
#voxel_size = 0.05
voxel_volume = voxel_grid
point_cloud_np = np.asarray([voxel_volume.origin + pt.grid_index*voxel_volume.voxel_size for pt in voxel_volume.get_voxels()])

#%%
x_points = point_cloud_np[:,0]
y_points = point_cloud_np[:,1]
z_points = point_cloud_np[:,2]

x_points -= min(x_points)
z_points -= min(z_points)
y_points -= min(y_points)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x_points, y_points, z_points)
plt.show()
#%%
pcd = point_cloud_np + np.array([1,1,1])
#%%
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
plt.plot(x_points, y_points, 'x')



#%%
"Set up initial grid"

def _rescale_2D_shape(polygons):

    #currently supports one filled body (body can be segmented)
    points_all = np.concatenate(([polygon for polygon in polygons]))
    xmin = min(points_all[:,0])
    xmax = max(points_all[:,0])
    ymin = min(points_all[:,1])
    ymax = max(points_all[:,1])
    Lx = abs(xmax - xmin)
    Ly = abs(ymax - ymin)
    
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
def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model


def voxel_carving(mesh,
                  output_filename,
                  camera_path,
                  cubic_size,
                  voxel_resolution,
                  w=300,
                  h=300,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    camera_sphere = o3d.io.read_triangle_mesh(camera_path)

    # setup dense voxel grid
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[1.0, 0.7, 0.0])

    # rescale geometry
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in enumerate(camera_sphere.vertices):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
        print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()

    # add voxel grid survace
    print('Surface voxel grid from %s' % surface_method)
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    else:
        raise Exception('invalid surface method')
    voxel_carving_surface = voxel_surface + voxel_carving

    return voxel_carving_surface, voxel_carving, voxel_surface
#%%

