import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir


    

def area(points):
    hull_2d = ConvexHull(points[:, :2])
    hull_area = hull_2d.volume
    return hull_area


def mean_height(points):
    mean_h = np.mean(points[:, 2])
    return mean_h


def eigenvalue_calculator(points, radius=0.2):
    kd_tree_3d = KDTree(points, leaf_size=5)

    neighbours = kd_tree_3d.query_radius(points, r=radius)

    cov = np.cov(neighbours.T)
    w, _ = np.linalg.eig(cov)
    w.sort()

    return w


def linearity(points):
    w = eigenvalue_calculator(points)
    return (w[2]-w[1]) / (w[2] + 1e-5)

def sphericity(points):
    w = eigenvalue_calculator(points)
    return w[0] / (w[2] + 1e-5)

def planarity(points):
    w = eigenvalue_calculator(points)
    return (w[1]-w[0]) / (w[2] + 1e-5)

def centre_of_mass(points):
    com = np.mean(points, axis=0)
    return np.mean(np.linalg.norm(points-com))

def mean_height(points):
    mean_h = np.mean(points[:, 2])
    return mean_h



def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points

def main():
    data_path = 'pointclouds-500/pointclouds-500'
    

if __name__=='__main__':
    main()



    # ####hella
    # hello
    # hello
    # hello
    # testing