"""plyファイルの視覚化を行う.

@author kawanoichi
実行コマンド
$ python3 show_ply.py
"""

import pyvista as pv
import argparse

def show_ply(path):
    # PLYファイルを読み込みます
    mesh = pv.read(path)

    # メッシュを表示します
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.show()



if __name__ == "__main__":    
    # ply_file =('predict_points/e50_p2048_airplane_01png.ply')

    print("終了")

    parser = argparse.ArgumentParser(description="使用例\n"
                                                 " 指定したplyファイルの視覚化する\n"
                                                 " $ python show_ply.py -m point_cloud.ply\n",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-m", "--mesh", type=str, help="視覚化する")
    args = parser.parse_args()

    # 指定していないときは'point_cloud.ply'のメッシュ化を行う
    if args.mesh is None:
        show_ply('point_cloud.ply')
        exit(0)

    show_ply(args.mesh)
