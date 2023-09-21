import numpy as np
import open3d as o3d

# PLYファイルの読み込み
file_path = "point_cloud"
# file_name = "Experiment_ply/regular_tetrahedron.ply"
print("file_path = ", file_path)
ply_file = o3d.io.read_point_cloud(file_path)

# 頂点座標の取得
vertices = np.asarray(ply_file.points)

# 面情報の取得
triangles = np.asarray(ply_file.triangles)

# 頂点と面から点群を生成
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(vertices)

# 面を描画する
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])
