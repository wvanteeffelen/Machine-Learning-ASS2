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

def main():
    pass
    

def area(points):
    hull_2d = ConvexHull(points[:, :2])
    hull_area = hull_2d.volume

    return hull_area


def mean_height(points):
    mean_h = np.mean(points[:, 2])
    return mean_h



if __name__=='__main__':
    main()