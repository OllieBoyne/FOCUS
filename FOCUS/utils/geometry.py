import numpy as np
import trimesh
from pytorch3d.structures import Meshes

def euler_to_rotation_matrix(euler):
	"""Given euler angles in degrees, return a rotation matrix"""
	roll, pitch, yaw = np.deg2rad(euler)
	R_x = np.array([
		[1, 0, 0],
		[0, np.cos(roll), -np.sin(roll)],
		[0, np.sin(roll), np.cos(roll)]
	])
	R_y = np.array([
		[np.cos(pitch), 0, np.sin(pitch)],
		[0, 1, 0],
		[-np.sin(pitch), 0, np.cos(pitch)]
	])
	R_z = np.array([
		[np.cos(yaw), -np.sin(yaw), 0],
		[np.sin(yaw), np.cos(yaw), 0],
		[0, 0, 1]
	])
	R = np.dot(R_z, np.dot(R_y, R_x))
	return R

def export_pointcloud_with_normals(points, normals, loc: str):
	"""
	points: N x 3 
	Normals: N x 3"""

	cloud = trimesh.PointCloud(vertices=points, normals=normals)
	cloud.export(loc)

def pytorch3d_meshes_to_trimesh(meshes: Meshes) -> trimesh.Trimesh:
	"""Convert a PyTorch3D Meshes object to a trimesh object
	Args:
		meshes: Meshes object
	Returns:
		trimesh object"""

	if len(meshes) > 1: raise ValueError("Only one mesh supported.")
	return trimesh.Trimesh(vertices=meshes.verts_packed().cpu().detach().numpy(),
                           faces = meshes.faces_packed().cpu().detach().numpy(),
						   process=False)