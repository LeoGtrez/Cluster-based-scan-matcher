import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from scipy.optimize import minimize
from Cluster_based_scan_matcher import *


if __name__ == '__main__':
    # set seed for reproducible results
    np.random.seed(12345)

    # create a set of points to be the reference
    xs = np.random.random_sample((50, 1))
    ys = np.random.random_sample((50, 1))
    reference_points = np.hstack((xs, ys))
    reference_points = np.array(reference_points)

    # transform the set of reference points to create a new set of
    # points for testing the Cluster-based implementation


    # 1. apply rotation to the new point set
    GT=np.array([0.5, 0.5, 10])
    
    theta = np.deg2rad(GT[2])
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s],
                    [s, c]])

    points_to_be_aligned = np.dot(rot, np.transpose(reference_points))

    # 2. apply translation to the new point set
    points_to_be_aligned += np.array([[GT[0]], [GT[1]]])
    points_to_be_aligned = np.transpose(points_to_be_aligned)
    
    Scan_A = pd.DataFrame(reference_points, columns =['X', 'Y'])
    Scan_B = pd.DataFrame(points_to_be_aligned, columns =['X', 'Y'])

    # run cluster-based-scan-matcher
    res,k,error = run_cluster(GT,Scan_A,Scan_B)
    
    print("Number of clusters used: ",k," clusters")
    print("Final solution: [Translation X[m]=", res.x[0], 
          " Translation Y[m]=", res.x[1], " Rotation angle[deg]=", res.x[2])
    print("Ground truth:", GT)
    print("Error with respect to ground truth:", error)
