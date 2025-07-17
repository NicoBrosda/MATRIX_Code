import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load STL file
mesh = trimesh.load(Path("/Users/nico_brosda/Cyrce_Messungen/3D_Files/misc-Cut002.stl"))

if not isinstance(mesh, trimesh.Trimesh):  # In case it's a Scene
    mesh = mesh.dump(concatenate=True)

# Convert to Pyrender mesh
pm_mesh = pyrender.Mesh.from_trimesh(mesh)

# Scene and camera setup
scene = pyrender.Scene()
scene.add(pm_mesh)

# Compute object bounds
center = mesh.bounds.mean(axis=0)
size = mesh.extents  # width, depth, height

# Create orthographic camera that fully covers the mesh
xmag = size[0] / 2 * 1.1  # add small margin
ymag = size[1] / 2 * 1.1
camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

# Position camera directly above the object, looking down the -Z axis
camera_pose = np.eye(4)
camera_pose[:3, 3] = center
camera_pose[2, 3] += size[2] + 10  # place camera above top of object

scene.add(camera, pose=camera_pose)

# Add light at same position
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)

# Render
r = pyrender.OffscreenRenderer(800, 800)
color, _ = r.render(scene)
plt.imsave(Path("/Users/nico_brosda/Cyrce_Messungen/3D_Files/misc-Cut002.png"), color)