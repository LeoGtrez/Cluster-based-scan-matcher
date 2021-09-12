
import numpy as np
import math
from sklearn_extra.cluster import KMedoids
from scipy.optimize import minimize

def kmedoids(k, Scan_A):
    n_clusters=k
    X = np.stack([Scan_A['X'].to_numpy()])
    X = np.transpose(X)
    Y = np.stack([Scan_A['Y'].to_numpy()])
    Y = np.transpose(Y)
    Scan_A_f_array = np.hstack((X,Y))
    kmedoids = KMedoids(n_clusters, metric='mahalanobis', method='pam').fit(Scan_A_f_array)
    medoids = np.array(kmedoids.cluster_centers_)
    medoids_T = []
    for i in range(0,len(medoids)):
        medoids_T.append(np.asarray([[medoids[i][0]],[medoids[i][1]]], dtype = np.float64))
    labels = np.array(kmedoids.labels_)
    return Scan_A_f_array, medoids, medoids_T, labels

def within_sum_of_squares(data, centroids, labels):
    """
    Compute sum of squares of a prototype clustering
    algorithm.
    """

    SSW = 0
    SSW_vector = []
    ssw_local = []
    for l in np.unique(labels):
        data_l = data[labels == l]
        resid = data_l - centroids[l]
        SSW = (resid**2).sum()
        SSW_label = [l,SSW]
        SSW_vector.append(SSW_label)
        ssw_local.append(SSW)
    for j in range(0,len(centroids)):
        if SSW_vector[j][1] < (max(ssw_local))/4:
            SSW_vector[j][1] = (max(ssw_local))/4
    return SSW_vector


def ScanB_to_qj(Scan_B):
    Xj = np.stack(Scan_B['X'].to_numpy())
    Yj = np.stack(Scan_B['Y'].to_numpy())

    qj=[]
    for i in range(0,len(Xj)):
        qj.append(np.asarray([[Xj[i]],[Yj[i]]], dtype=np.float64))
    return qj

def transformation (X, qj):
    pj = []
    trl = np.array([[X[0]],[X[1]]])
    cos, sin = math.cos(X[2]), math.sin(X[2])
    rot = np.array([[cos, -sin],
                    [sin, cos]])
    for i in range (0, len(qj)):
        temp = np.dot(rot, qj[i])
        pj.append(temp + trl)
    pj = np.array(pj)
    return pj, cos, sin

def score_function(X):
    pj, cos, sin  = transformation(X, qj)
    S = 0
    for l in range(0,len(medoids_T)):
        for i in range(0,len(pj)):
            Alpha = pj[i] - medoids_T[l]
            Alpha_T = np.transpose(Alpha)
            S = S - np.exp(np.matmul(-Alpha_T,(1/SSW_vector[l][1])*Alpha))
    return S[0][0]

def score_function_2(X):
    pj, cos, sin  = transformation(X, qj)
    S = 0
    for l in np.unique(labels):
        pj_l = pj[labels == l]
        for i in range(0,len(pj_l)):
            Alpha = pj_l[i] - medoids_T[l]
            Alpha_T = np.transpose(Alpha)
            S = S - np.exp(np.matmul(-Alpha_T,(1/SSW_vector[l][1])*Alpha))
    return S[0][0]

def gradient_vector(X):
    pj, cos, sin  = transformation(X, qj)
    gx = [0,0,0]
    for l in range(0,len(medoids_T)):
        for i in range(0,len(pj)):
            Alpha = pj[i] - medoids_T[l]
            Alpha_T = np.transpose(Alpha)
            dAlpha_dX = np.asarray([[1, 0, -qj[i][0]*sin-qj[i][1]*cos],
                                 [0, 1, qj[i][0]*cos-qj[i][1]*sin]], dtype=np.float64)
            A = (2*np.exp(np.matmul(-Alpha_T,(1/SSW_vector[l][1])*Alpha)))
            B = np.matmul(Alpha_T,(1/SSW_vector[l][1])*dAlpha_dX)
            gx_new = A*B
            gx = gx + gx_new
    return gx[0]

def gradient_vector_2(X):
    pj, cos, sin  = transformation(X, qj)
    qj_arr = np.array(qj)
    gx = [0,0,0]
    for l in np.unique(labels):
        pj_l = pj[labels == l]
        qj_l = qj_arr[labels == l]
        for i in range(0,len(pj_l)):
            Alpha = pj_l[i] - medoids_T[l]
            Alpha_T = np.transpose(Alpha)
            dAlpha_dX = np.asarray([[1, 0, -qj_l[i][0]*sin-qj_l[i][1]*cos],
                                 [0, 1, qj_l[i][0]*cos-qj_l[i][1]*sin]], dtype=np.float64)
            A = (2*np.exp(np.matmul(-Alpha_T,(1/SSW_vector[l][1])*Alpha)))
            B = np.matmul(Alpha_T,(1/SSW_vector[l][1])*dAlpha_dX)
            gx_new = A*B
            gx = gx + gx_new
    return gx[0]

def column(matrix, i):
    return [row[i] for row in matrix]

def hessian_matrix (X):
    H = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
    pj, cos, sin  = transformation(X, qj)
    for l in range(0,len(medoids_T)):
        for i in range(0, len(pj)):
            Alpha = pj[i] - medoids_T[l]
            Alpha_T = np.transpose(Alpha)
            dAlpha_dX = np.asarray([[1, 0, -qj[i][0]*sin-qj[i][1]*cos],
                                 [0, 1, qj[i][0]*cos-qj[i][1]*sin]], dtype=np.float64)
            d2Alpha_dX2dX2 = np.asarray([-qj[i][0]*cos+qj[i][1]*sin,
                                      -qj[i][0]*sin-qj[i][1]*cos], dtype=np.float64)
    
            C = 2*np.exp(np.matmul(-Alpha_T, (1/SSW_vector[l][1])*Alpha))
            
            H_00 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))+
                      np.matmul(column(dAlpha_dX, 0), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 0)))))
            
            H_01 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))+
                      np.matmul(column(dAlpha_dX, 1), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 0)))))
            
            H_02 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))+
                      np.matmul(column(dAlpha_dX, 2), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 0)))))
            
            H_10 = H_01
            
            H_11 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))+
                      np.matmul(column(dAlpha_dX, 1), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 1)))))
            
            H_12 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))+
                      np.matmul(column(dAlpha_dX, 2), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 1)))))
            
            H_20 = H_02
            
            H_21 = H_12
            
            H_22 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))+
                      np.matmul(column(dAlpha_dX, 2), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 2))) + np.matmul(Alpha_T, (1/SSW_vector[l][1])*
                                                                             d2Alpha_dX2dX2)))
            
            H_new = np.array([[H_00[0][0], H_01[0][0], H_02[0][0]],
                              [H_10[0][0], H_11[0][0], H_12[0][0]],
                              [H_20[0][0], H_21[0][0], H_22[0][0]]])
            
            H = H + H_new
            
    return H

def hessian_matrix_2 (X):
    H = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
    pj, cos, sin  = transformation(X, qj)
    qj_arr = np.array(qj)
    for l in np.unique(labels):
        pj_l = pj[labels == l]
        qj_l = qj_arr[labels == l]
        for i in range(0, len(pj_l)):
            Alpha = pj_l[i] - medoids_T[l]
            Alpha_T = np.transpose(Alpha)
            dAlpha_dX = np.asarray([[1, 0, -qj_l[i][0]*sin-qj_l[i][1]*cos],
                                 [0, 1, qj_l[i][0]*cos-qj_l[i][1]*sin]], dtype=np.float64)
            d2Alpha_dX2dX2 = np.asarray([-qj_l[i][0]*cos+qj_l[i][1]*sin,
                                      -qj_l[i][0]*sin-qj_l[i][1]*cos], dtype=np.float64)
    
            C = 2*np.exp(np.matmul(-Alpha_T, (1/SSW_vector[l][1])*Alpha))
            
            H_00 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))+
                      np.matmul(column(dAlpha_dX, 0), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 0)))))
            
            H_01 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))+
                      np.matmul(column(dAlpha_dX, 1), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 0)))))
            
            H_02 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,0)))+
                      np.matmul(column(dAlpha_dX, 2), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 0)))))
            
            H_10 = H_01
            
            H_11 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))+
                      np.matmul(column(dAlpha_dX, 1), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 1)))))
            
            H_12 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,1)))+
                      np.matmul(column(dAlpha_dX, 2), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 1)))))
            
            H_20 = H_02
            
            H_21 = H_12
            
            H_22 = (C*(-2*np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))*
                      np.matmul(Alpha_T, (1/SSW_vector[l][1])*np.transpose(column(dAlpha_dX,2)))+
                      np.matmul(column(dAlpha_dX, 2), (1/SSW_vector[l][1])*
                                np.transpose(column(dAlpha_dX, 2))) + np.matmul(Alpha_T, (1/SSW_vector[l][1])*
                                                                             d2Alpha_dX2dX2)))
            
            H_new = np.array([[H_00[0][0], H_01[0][0], H_02[0][0]],
                              [H_10[0][0], H_11[0][0], H_12[0][0]],
                              [H_20[0][0], H_21[0][0], H_22[0][0]]])
            
            H = H + H_new
            
    return H

def CSOG_fixed(X0, ground_truth, Scan_A):
    k = 20 #Number of initial clusters
    global labels, medoids_T, SSW_vector
    #Apply medoids to obtain Scan_A in array shape, medoids and labels
    Scan_A_f_array, medoids, medoids_T, labels = kmedoids(k, Scan_A)
    #Obtain WSS "Within-sum-of-squares" 
    SSW_vector = within_sum_of_squares(Scan_A_f_array, medoids, labels)
    res = minimize(score_function_2, X0, method='trust-ncg',
                   jac=gradient_vector_2, hess=hessian_matrix_2,
                   options={'maxiter': 1000, 'disp': False})

    return res, k

def CSOG_iter(X0, ground_truth, Scan_A):
    it=1
    k = 20 #Number of initial clusters
    tol = 0.1 #Convergence criteria
    res_old = np.zeros((3,))
    global labels, medoids_T, SSW_vector
    while (tol > 1e-8):
        #Apply medoids to obtain Scan_A in array shape, medoids and labels
        Scan_A_f_array, medoids, medoids_T, labels = kmedoids(k, Scan_A)
        #Obtain WSS "Within-sum-of-squares"
        SSW_vector = within_sum_of_squares(Scan_A_f_array, medoids, labels)
        res = minimize(score_function_2, X0, method='trust-ncg',
                   jac=gradient_vector_2, hess=hessian_matrix_2,
                   options={'maxiter': 1000, 'disp': False})
        tol = (res.x - res_old).sum()
        if tol > 1e-8:
            if (abs(len(medoids) - len(Scan_A)) >= 10):
                k += 10
            else:
                break
        res_old = res.x
        X0 = res.x
    return res, k

def run_cluster(GT,Scan_A, Scan_B):
    # run cluster-based-scan-matcher
    X0 = np.array([0,0,0]) #Initial x,y translation + rotation for optimization algorithm (minimize)
    global qj
    qj = ScanB_to_qj(Scan_B)
    #Optimization algorithm
    res,k = CSOG_iter(X0, GT, Scan_A)
    res.x[2] = math.degrees(res.x[2])
    error = abs(res.x - (-GT))
    return res,k,error
    
