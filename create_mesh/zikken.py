import open3d as o3d
import os
import numpy as np

from information import SCRIPT_DIR_PATH, PROJECT_DIR_PATH, PLY_DIR_PATH



if __name__=="__main__":
    print("実験開始！！")

    ply_file = "point_cloud.ply"
    ply_file_path = os.path.join(PLY_DIR_PATH, ply_file)
    output_ply_path = os.path.join(PLY_DIR_PATH, "marching_"+ply_file)
    # main(ply_file_path, output_ply_path)

    ptCloud = o3d.io.read_point_cloud(ply_file_path)
    # >>> PointCloud with 8 points.

    print(np.asarray(ptCloud.points))

    print(ptCloud)
    print("終了")