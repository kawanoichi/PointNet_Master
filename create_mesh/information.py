"""パス情報を定義するファイル.

from information import SCRIPT_DIR_PATH, PROJECT_DIR_PATH, PLY_DIR_PATH
"""

import os

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "ply_data")


if __name__=="__main__":
    print("SCRIPT_DIR_PATH  :", SCRIPT_DIR_PATH)
    print("PROJECT_DIR_PATH :", PROJECT_DIR_PATH)
    print("PLY_DIR_PATH     :", PLY_DIR_PATH)