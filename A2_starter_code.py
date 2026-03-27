"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm 
"""

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
from sklearn.ensemble import RandomForestClassifier as RF



def get_feature_indices(selected_feature_names):
    feature_names = ["std_height", "dx", "dy", "dz",
                     "3d_density", "root_density", "hull_area", "shape_index",
                     "circularity", "linearity", "sphericity", "planarity"]
    return [feature_names.index(name) for name in selected_feature_names]



class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        # 1. calculate standard deviation of the height
        std_height = np.std(self.points[:, 2])
        self.feature.append(std_height)

        # 2. calculate the 3D bounding box dimensions
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        dx,dy,dz = maxs - mins
        self.feature.extend([dx,dy,dz])

        # 3. calculate the 3D density
        volume = dx * dy * dz + 1e-5
        density_3d = len(self.points) / volume
        self.feature.append(density_3d)

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # 4. compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)

        # 5. Hull Area - compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        hull_perimeter = hull_2d.area
        self.feature.append(hull_area)

        # 6. Shape Index - get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)

        # 7. Circularity - determine the circularity
        circularity = 4 * math.pi * hull_area / (hull_perimeter**2 + 1e-5)
        self.feature.append(circularity)
    
        # obtain the point cluster near the top area
        k_top = max(int(len(self.points) * 0.005), 100)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        # obtain the covariance matrix of the top points
        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)
        w.sort()

        # 8,9,10. calculate the linearity, sphericity and planarity
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        planarity = (w[1]-w[0]) / (w[2] + 1e-5)
        self.feature += [linearity, sphericity, planarity]

def feature_selection(X, y):
    # Within-class scatter matrix:
    N = len(X)
    mean = np.mean(X, axis=0)
    classes = np.unique(y)
    cov_matrices = {}
    total_within = 0
    total_between = 0
    for k in classes:
        Xk = X[y == k]
        Nk = len(Xk)
        mean_k = np.mean(Xk, axis=0)

        Xk_diff = Xk - mean_k
        cov_k = (Xk_diff.T @ Xk_diff) / (Nk - 1)
        cov_matrices[k] = cov_k

        within = (Nk/N)*cov_k
        total_within += within

        mean_k = np.mean(Xk, axis=0)
        diff = (mean_k - mean).reshape(-1, 1) 
        between = (Nk/N)*(diff @ diff.T)
        total_between += between
    
    J = np.trace(total_between)/np.trace(total_within)

    return J

def select_4_features():
    feature_names = ["std_height", "dx", "dy", "dz", "3d_density", "root_density", "hull_area","shape_index", "circularity", "linearity", "sphericity", "planarity"]
    scores = []

    for i in range(0, X.shape[1]):
        J = feature_selection(X[:, i:i+1], y)
        scores.append((feature_names[i], J))

    scores.sort(key=lambda x: x[1], reverse=True)

    for name, score in scores:
        print(f"{name}: {score:.4f}")

    top4 = scores[:4]
    top4_names = []
    for name, score in top4:
        top4_names.append(name)

    print(f'The four features with the best score are: {top4_names}')
    return top4_names

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


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = data_header = 'ID,label,std_height,dx,dy,dz,3d_density,root_density,hull_area,shape_index,circularity,linearity,sphericity,planarity'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)



def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']
    
    maxF = max(X[:,0])
    minF = min(X[:,0])

    bins = np.linspace(minF, maxF, 11)  # 10 bins

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for i in range(5):
        data = X[100*i:100*(i+1), 0]
        axes[i].hist(data, bins=bins, color=colors[i])
        axes[i].set_title(f"{labels[i]}")
    fig.suptitle(f"Histograms of the feature x", fontsize=16)
    plt.tight_layout()
    plt.show()


def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    clf = RF(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("RF accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, pred)
    print(conf)


if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = 'pointclouds-500/pointclouds-500'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # determine the 4 most important features
    print('Feature importance based on Within-class scatter matrix and Between-class scatter matrix...')
    selected_features = select_4_features()

    # visualize features
    print('Visualize the features')
    feature_visualization(X=X)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # RF classification
    print('Start RF classification')
    RF_classification(X, y)

    # select 4 best features
    selected_indices = get_feature_indices(selected_features)
    X_selected = X[:, selected_indices]

    SVM_classification(X_selected, y)
    RF_classification(X_selected, y)