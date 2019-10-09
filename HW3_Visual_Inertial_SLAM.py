import numpy as np
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm
from numpy.linalg import inv,pinv
from numpy.random import normal

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"] # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      K = data["K"] # intrindic calibration matrix
      b = data["b"] # baseline
      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
  return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu


def visualize_trajectory_2d(pose,landmark,path_name="Unknown",show_ori=True):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  if show_ori:
      select_ori_index = list(range(0,n_pose,int(n_pose/50)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  ax.scatter(landmark[:,0],landmark[:,1],color='green',marker='+')
  plt.show(block=True)
  return fig, ax


if __name__ == '__main__':
    filename = "./data/0020.npz"
    dataset = os.path.splitext(os.path.basename(filename))[0]
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    #stereo camera calibration matrix M
    fsu,fsv,cu,cv = K[0,0],K[1,1],K[0,2],K[1,2]
    M = np.array([[fsu,0,cu,0],[0,fsv,cv,0],[fsu,0,cu,-fsu*b],[0,fsv,cv,0]])
    canon = np.hstack((np.identity(3),np.zeros([3,1])))
    #time step
    tau = []
    for i in range(t.shape[1]-1):
        tau.append(t[0,i+1]-t[0,i])
    tau = np.array(tau)
    def pi(q):
        pi_q = q/q[2]
        return pi_q
    
    def inv_pi(q):
        inv_pi = q[2]*q
        return inv_pi
    
    
    def dpi_dq(q):
        mat = np.zeros([4,4])
        mat[0,:] = np.array([1,0,-q[0]/q[2],0])
        mat[1,:] = np.array([0,1,-q[1]/q[2],0])
        mat[3,:] = np.array([0,1,-q[3]/q[2],1])
        dpidq =1/q[2]*mat
        return dpidq
    
    def hat_map(X):
        X_hat = np.zeros([3,3])
        X_hat[0,:] = np.array([0,-X[2],X[1]])
        X_hat[1,:] = np.array([X[2],0,-X[0]])
        X_hat[2,:] = np.array([-X[1],X[0],0])
        return X_hat
    def mexp(X):
        exp = np.identity(X.shape[0])+ X +1/2*X.dot(X) + 1/6*X.dot(X.dot(X)) + 1/24*X.dot(X.dot(X.dot(X)))
        return exp
    
    #def homo_coord
	# (a) IMU Localization via EKF Prediction
    def prediction(mu,sigma,v,omega,tau):
        o_hat = hat_map(omega)
        v_hat = hat_map(v)
        v_t = np.array([v[0],v[1],v[2]]).T
        u_hat = np.vstack((np.vstack((o_hat.T,v_t.T)).T,[0,0,0,0]))
        u_hhat = np.vstack((np.vstack((o_hat.T,v_hat.T)).T,np.vstack((np.zeros([3,3]),o_hat.T)).T))
        mu_pred = mexp(-tau*u_hat).dot(mu)
        sigma_pred = mexp(-tau*u_hhat).dot(sigma.dot(mexp(tau*u_hhat).T)) + tau*tau*W
        return mu_pred,sigma_pred
	# (b) Landmark Mapping via EKF Update
    def mapping(T_oi,T_t,m_t,z_t):
        valid = np.logical_not(z_t == -1)
        corres_idx = np.where(valid[0] == True)[0]
        pt_num = mu_map.shape[1]
        for idx in corres_idx:
            z_ti = z_t[:,idx]
            uLvL = np.append(z_ti[0:2],[1])
            uL,vL,uR,vR= z_ti[0],z_ti[1],z_ti[2],z_ti[3]
            depth = (fsu*b)/(uL-uR)
            xyz_O = np.append(depth*inv(K).dot(uLvL),1)
            xyz_W = T_t.dot(inv(T_oi).dot(xyz_O))
            m_t[:,idx] = xyz_W
        return m_t
    
    def update_map(T_oi,T_t,z_t,mu_map,sigma_m):
        V = 1* np.identity(4)
        z_hat = -np.ones(z_t.shape)
        K_t =np.zeros([3,4,z_t.shape[1]])
        H_t =np.zeros([4,3,z_t.shape[1]])
        
        valid = np.logical_not(z_t == -1)
        corres_idx = np.where(valid[0] == True)[0]
        mu_init = -np.ones([z_t.shape[0],z_t.shape[1]])
        #z_t = z_t[:,valid[1]]
        TT= T_oi.dot(T_t)
        TTmu = T_oi.dot(T_t.dot(mu_map))
        for idx in corres_idx:
            H_t[:,:,idx] = M.dot(dpi_dq(TTmu[:,idx])).dot(D)
            z_hat[:,idx] = M.dot(pi(TTmu[:,idx]))
            mu_init[:,idx] = mu_map[:,idx]
        mu_map = mu_init
        for j in range(z_t.shape[1]):
            K_t[:,:,j] = sigma_m[:,:,j].dot(H_t[:,:,j].T).dot(inv(H_t[:,:,j].dot(sigma_m[:,:,j]).dot(H_t[:,:,j].T)+V))
            mu_map[:,j] = mu_map[:,j] + D.dot(K_t[:,:,j]).dot(z_t[:,j]-z_hat[:,j])
            sigma_m[:,:,j] = (np.identity(3) - K_t[:,:,j].dot(H_t[:,:,j])).dot(sigma_m[:,:,j]) 
        return mu_map,sigma_m
    
	# (c) Visual-Inertial SLAM (Extra Credit)
    
    def update_SLAM(T_oi,T_t,z_t,mu,sigma,m_t):
        V = 1* np.identity(4)
        z_hat = -np.ones(z_t.shape)
        
        valid = np.logical_not(z_t == -1)
        corres_idx = np.where(valid[0] == True)[0]
        K_t =np.zeros([3,4,z_t.shape[1]])
        H_t =np.zeros([4,6,z_t.shape[1]])


        
        return 
        
    mu = np.identity(4)
    sigma = np.zeros([6,6])
    
    W = np.identity(6)
    traj = np.array([[0,0]])
    pose = np.zeros([4,4,t.shape[1]])
    pose[0:4,0:4,0] = np.identity(4)
    landmark = np.array([[0,0,0,1]])
    T_oi = cam_T_imu
    #initialize mu_map,sig_map
    mu_map = features[:,:,0]
    m_t = np.zeros([3,mu_map.shape[1]])
    m_t = np.vstack((m_t,np.ones(mu_map.shape[1])))
    sigma_m = np.zeros([3,3,features.shape[1]])
    for i in range(features.shape[1]):
        sigma_m[:,:,i] = np.identity(3)
    
    D = np.vstack((np.identity(3),np.zeros([1,3])))
    for i in range(len(tau)):
        omega = rotational_velocity[:,i+1]
        v = linear_velocity[:,i+1]
        #(1) prediction
        mu,sigma = prediction(mu,sigma,v,omega,tau[i])
        T_t = inv(mu)
        z_t = features[:,:,i+1]
        #pose = np.vstack((pose,Tt))
        pose[0:4,0:4,i+1] = T_t
        #(2) mapping EKF update
        mu_map,sigma_m = update_map(T_oi,T_t, z_t,mu_map,sigma_m)
        m_t = mapping(T_oi,T_t,m_t,mu_map)
        #m_t = mapping(T_oi,T_t,m_t,z_t)
        #(3) VI-SLAM update
        
        x,y,z = T_t[0,3],T_t[1,3],T_t[2,3]
        traj=np.vstack((traj,np.array([x,y])))
    
    xvalid = np.logical_and((m_t[0,:]>=-200),(m_t[0,:]<=400))
    yvalid = np.logical_and((m_t[1,:]>=-100),(m_t[1,:]<=100))
    mvalid = np.logical_and(xvalid,yvalid)
    m_t = m_t[:,mvalid]
    landmark = m_t.T
    fig = plt.figure()
    
    fig,ax = visualize_trajectory_2d(pose,landmark, path_name = "trajectory"+dataset)
    
    plt.show()
	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
