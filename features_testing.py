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
        # calculate standard deviation of the height
        std_height = np.std(self.points[:, 2])
        self.feature.append(std_height)

        # calculate the 3D bounding box dimensions
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        dx,dy,dz = maxs - mins
        self.feature.extend([dx,dy,dz])

        # calculate the 3D density
        volume = dx * dy * dz + 1e-5
        density_3d = len(self.points) / volume
        self.feature.append(density_3d)

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)

        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        hull_perimeter = hull_2d.area
        self.feature.append(hull_area)

        # get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)

        # determine the circularity
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

        # calculate the linearity, sphericity and planarity
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        planarity = (w[1]-w[0]) / (w[2] + 1e-5)
        self.feature += [linearity, sphericity, planarity]


# center of mass, calculate spread around the centroid
        com = np.mean(self.points, axis=0)
        spread_com = np.mean(np.linalg.norm(self.points-com))
        self.feature.append(spread_com)


# calculate the height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)


# calculate the mean height
        mean_h = np.mean(self.points[:, 2])
        self.feature.append(mean_h)