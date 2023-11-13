import open3d as o3d

def ball_pivoting(point_cloud):
    # 法線の計算
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Ball-Pivotingアルゴリズムを使用して面を構築
    triangles = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector([0.02, 0.1]))

    return triangles

# 仮の点群データの生成
point_cloud = o3d.geometry.PointCloud()
points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]]
point_cloud.points = o3d.utility.Vector3dVector(points)

# 法線の計算とBall-Pivotingアルゴリズムの実行
result_mesh = ball_pivoting(point_cloud)

print("result_mesh", result_mesh)

# 結果の表示
o3d.visualization.draw_geometries([point_cloud, result_mesh])
