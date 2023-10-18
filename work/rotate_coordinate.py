"""点群座標を回転させるモジュール."""
import numpy as np


def rotate_around_y_axis(point, angle_degrees, reverse=False):
    angle_radians = np.radians(angle_degrees)
    direction = -1 if reverse else 1
    rotation_matrix = np.array([
        [np.cos(direction * angle_radians), 0,
         np.sin(direction * angle_radians)],
        [0, 1, 0],
        [-np.sin(direction * angle_radians), 0,
         np.cos(direction * angle_radians)]
    ])
    rotated_point = np.dot(rotation_matrix, point)
    return rotated_point


def rotate_around_x_axis(point, angle_degrees, reverse=False):
    angle_radians = np.radians(angle_degrees)
    direction = -1 if reverse else 1
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(direction * angle_radians), -
         np.sin(direction * angle_radians)],
        [0, np.sin(direction * angle_radians),
         np.cos(direction * angle_radians)]
    ])
    rotated_point = np.dot(rotation_matrix, point)
    return rotated_point
