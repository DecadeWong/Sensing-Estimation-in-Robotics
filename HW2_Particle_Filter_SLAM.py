import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from numpy.random import normal
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
import time
import cv2
import os
import blend

def logOdds_to_prob(logOdds):
    p = 1 - 1/(1 + np.exp(logOdds))
    return p

def lowpass_filter(data, dt, fc):
    imu = data
    RC = 1/(2*np.pi*fc)
    a = dt/(RC+dt)
    yaw_rate = [0]
    yaw_rate[0] = a*data[0]
    for x in data:
        yaw_rate.append((1-a)*yaw_rate[-1]+a*x)
    return yaw_rate



#origin of world frame is defined at the rear axle of the robot at time 0
def lidar2body(ranges):
    angles = np.arange(45,315.25,0.25)*np.pi/180.0
    #angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    x0 = ranges*np.cos(angles)
    y0 = ranges*np.sin(angles)
    T_l2b = [[1,0,-0.29833], [0,1,0], [0,0,1]]
    x1, y1, theta1= [], [], []
    theta1 = angles
    for i in range(len(ranges)):
        x1.append(np.dot(T_l2b, [x0[i],y0[i],1])[0])
        y1.append(np.dot(T_l2b, [x0[i],y0[i],1])[1])
    x = np.vstack((x1,y1,theta1))
    return x


def body2world(x, y, theta, ts):
    for i in range(ts):
        FR,FL,RR,RL = encoder_counts[0,i],encoder_counts[1,i],encoder_counts[2,i],encoder_counts[3,i]
        V_R = (FR+RR)/2*0.0022/tau[i]
        V_L = (FL+RL)/2*0.0022/tau[i]
        V = (V_R+V_L)/2
        theta = theta + omega[i]*tau[i]
        x = x + tau[i]*V*np.sinc(omega[i]*tau[i]/2)*np.cos(theta+omega[i]*tau[i]/2)
        y = y + tau[i]*V*np.sinc(omega[i]*tau[i]/2)*np.sin(theta+omega[i]*tau[i]/2)
    return x, y, theta
        

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))

def mapping(MAP_input, ranges, T_bl, x_t):
    gamma = 0.8 # occupancy probability
    x = x_t[0]
    y = x_t[1]
    theta = x_t[2]
    MAP = np.copy(MAP_input)
    #body to world transformation
    T_wb = np.zeros([4,4])
    T_wb[0,:] = [np.cos(theta), -np.sin(theta), 0, x]
    T_wb[1,:] = [np.sin(theta),  np.cos(theta), 0, y]
    T_wb[2,2], T_wb[3,3] =0 , 0
    
    #turn the lider 180 degree ccw to align the angles with world frame
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    
     # xy position in the lidar frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    
    #observation in lidar frame
    z_l = np.vstack((xs0,ys0,np.zeros(xs0.size),np.ones(xs0.size)))
    
    z_b = np.dot(T_bl, z_l)
    z_t = np.dot(T_wb, z_b)
    #z_t = z_t[:, zs_world[2, :] > -0.92]
    m_ratio = 10
    #set center of the map at (10,5)
    ex = np.round((z_t[0,:]-10)*m_ratio+300).astype('int')
    ey = np.round(300-(z_t[1,:]-5)*m_ratio).astype('int')
    sx = np.round((x-10)*m_ratio+300).astype('int')
    sy = np.round(300-(y-5)*m_ratio).astype('int')
    
    for i in range(z_t.shape[1]):
        grids = bresenham2D(sx,sy,ex[i],ey[i]).astype('int16')
        Valid = np.logical_and(np.logical_and(grids[0,:]>=0, grids[0,:]<=600),np.logical_and(grids[1,:]>=0, grids[1,:]<=600) )
        grids = grids[:,Valid]
        MAP[grids[1,:-1],grids[0,:-1]] += np.log((1-gamma)/gamma)#free cells
        MAP[grids[1,-1],grids[0,-1]] += np.log(gamma/(1-gamma))#occupied cells
    return MAP


def resample(mu, alpha):
    mu_res = []
    j = 0
    c = alpha[0]
    N = len(alpha)
    alpha_res = 1/N * np.ones(N)
    mu = mu.T
    for k in range(N):
        u = 1/(2*N)
        beta = u + k/N
        while beta > c:
            j += 1
            c += alpha[j]
        mu_res.append(mu[j])
    mu_res=np.array(mu_res).T
    return mu_res, alpha_res

def binary_map(log_map):
    prob_map = logOdds_to_prob(log_map)
    MAP_0 = np.zeros([600, 600])
    for i in range(MAP_0.shape[0]):
        for j in range(MAP_0.shape[1]):
            if prob_map[i,j] >= 0.8:
                MAP_0[i,j] = 1
            elif prob_map[i,j] <=0.2:
                MAP_0[i,j] = -1
            else:
                MAP_0[i,j] = 0
    return MAP_0

def plot_map(MAP,trajectory):
    MAP = -binary_map(MAP)
    #plot trajectory
    trajectory = np.array(trajectory)  
    fig = plt.figure(figsize=(10,10))
            
    plt.imshow(MAP,cmap="gray") 
    plt.scatter(trajectory[:,0],trajectory[:,1],s=1,c='b')
    plt.show()


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  #nx = im.shape[0]
  #ny = im.shape[1]
  ratio = 10
  img_size = im.shape[0]
  #xmin = x_im[0]
  #xmax = x_im[-1]
  #xresolution = 0.1
  #ymin = y_im[0]
  #ymax = y_im[-1]
  #yresolution = 0.1
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    #iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      #ix = np.int16(np.round((x1-xmin)/xresolution))
      x_map = np.round(x1*ratio + img_size/2).astype(int)
      y_map = np.round(img_size/2 - y1*ratio).astype(int)
      valid = np.logical_and( np.logical_and((x_map >=0), (x_map < img_size)), \
			                        np.logical_and((y_map >=0), (y_map < img_size)))
      cpr[jy,jx] = np.sum(im[y_map[valid],x_map[valid]])
  return cpr

#grid size 1/m_ratio = 0.1 m

def update(mu,alpha,ranges, log_map, T_bl):
    N = len(alpha)
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # xy position in the lidar frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    
    #observation in lidar frame
    z_l = np.vstack((xs0,ys0,np.zeros(xs0.size),np.ones(xs0.size)))
    #transfer to body frame
    z_b = np.dot(T_bl, z_l)
    x_im = np.arange(-30.0,50.1,.1)
    y_im = np.arange(-35.0,45.1,.1)
    alpha_update = np.zeros(N)
    MAP = binary_map(log_map)
    mu_var = mu
    
    #var_theta = np.array([-.4,-.3,-.2,-.1,0,.1,.2,.3,.4])
    mu_update = np.copy(mu)
    for i in range(N):
        #cpr_set = []
        #for j in range(len(var_theta)):
        mu_var[:,i] = mu[:,i] #+ normal(0,0.01,3)
        x,y,theta = mu_var[0,i], mu_var[1,i], mu_var[2,i]
        
        #theta = theta + var_theta[j]
        T_wb = np.zeros([4,4])
        T_wb[0,:] = [np.cos(theta), -np.sin(theta), 0, x]
        T_wb[1,:] = [np.sin(theta),  np.cos(theta), 0, y]
        T_wb[2,2], T_wb[3,3] =0 , 0
        z_t1 = np.dot(T_wb, z_b)
            
        cpr = mapCorrelation(MAP,x_im,y_im,z_t1,xs,ys)
        #cpr_set.append(cpr)
        #cpr_set =np.array(cpr_set)
        
        
        alpha_update[i] = alpha[i]*np.exp(cpr.max())
    #normalize
    norm = np.linalg.norm(alpha_update,ord =1)
    alpha_update = alpha_update/norm
    return mu_update,alpha_update

#texture mapping 
def texture(uv, disp, rgb, x_t, tmap):
    theta =  x_t[-1]
    R_wb = np.zeros([3,3])
    R_wb[0,:] = [np.cos(theta), -np.sin(theta), 0]
    R_wb[1,:] = [np.sin(theta),  np.cos(theta), 0]
    R_wb[2,2] = 1
    p_wb = x_t[0:2].copy().reshape(2,1)
    p_wb = np.row_stack((p_wb,np.array([0.127])))
    p_wc = p_wb +p_bc
    R_wc = R_wb.dot(R_bc)
    disparity = disp.reshape(1,disp.size)
    dd = -0.00304*disparity + 3.31
    depth = 1.03/dd
    xyz_0 = K_inv.dot(uv) * depth
    xyz_w = R_wc.dot(R_oc.T).dot(xyz_0+R_oc.dot(R_wc.T).dot(p_wc))
    idx_floor = np.where(xyz_w[2] < 0.2)
    disp_floor = disparity[:,idx_floor[0]]
    u_floor = uv[0,idx_floor[0]]
    v_floor = uv[1,idx_floor[0]]
    dd1 = -0.00304*disp_floor + 3.31
    rgbi = (u_floor*526.37 + dd1 * (-4.5*1750.46) +19276.0)/585.051
    rgbj = (v_floor*526.37 + 16662.0)/585.051
    rgbi = rgbi.astype(np.int16)
    rgbj = rgbj.astype(np.int16)
    #find floor grid postion in the world frame
    floor = xyz_w[:,idx_floor[0]]
    x_map = np.round((floor[0]-10)*m_ratio + img_size/2).astype(int)
    y_map = np.round(img_size/2 - (floor[1]-5)*m_ratio).astype(int)
    valid = np.logical_and(np.logical_and(x_map>0, x_map<img_size),np.logical_and((y_map>0), (y_map<img_size))) 
    tmap[y_map[valid[0]],x_map[valid[0]]] = rgb[rgbj[valid[0]],rgbi[valid[0]]]
    tmap = tmap.astype(np.uint8)
    return tmap
    
data = imu_angular_velocity[2,:]
yaw_rate = lowpass_filter(data, 0.01, 10)
yaw_rate.pop(0)
yaw_rate_sync = []
#lidar_range_sync = np.zeros((lidar_ranges.shape[0],len(encoder_stamps)+1))
lidar = []

#synchronize encoder readings， lidar scan and yaw_rate
for t in encoder_stamps:
    diff =[]
    for i in range(len(imu_stamps)):
        diff.append(np.absolute(t-imu_stamps[i]))
    val,idx = min((val,idx) for (idx,val) in enumerate(diff))
    yaw_rate_sync.append(yaw_rate[idx])
    diff = []
    for j in range(len(lidar_stamps)):
        diff.append(np.absolute(t-lidar_stamps[j]))
    val,idx = min((val,idx) for (idx,val) in enumerate(diff))
    #lidar_range_sync[:,i+1] = lidar_ranges[:,idx]
    lidar.append(lidar_ranges[:,idx])
plt.plot(yaw_rate_sync)

for t in disp_stamps:
    diff =[]

omega = yaw_rate_sync
lidar = np.vstack(lidar).T
tau =[]
for i in range(len(encoder_stamps)-1):
    step= encoder_stamps[i+1] - encoder_stamps[i]
    tau.append(step)

#process and synchronize data
#parameters
m_ratio = 10
img_size = 600
Disparity ="Disparity20"
RGB = "RGB20"
xs = np.array([-.4,-.3,-.2,-.1,0,.1,.2,.3,.4])
ys = np.array([-.4,-.3,-.2,-.1,0,.1,.2,.3,.4])
#global transformation matrices
#transformation matrix from lidar to body 
T_bl = [[1,0,0,-0.29833], [0,1,0,0],[0,0,1,0],[0,0,0,1]]
R_oc = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
p_bc = np.array([[-.33276],[0],[.38001]])
roll,pitch,yaw = 0,0.36,0.021
R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw),  np.cos(yaw), 0],[0,0,1]])
R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
R_x = np.array([[1,0,0], [0, np.cos(roll), -np.sin(roll)],[0, np.sin(roll),  np.cos(roll)]])
R_bc = R_z.dot(R_y.dot(R_x))
K = np.array([[585.05108211, 0, 242.94140713],[0, 585.05108211, 315.83800193],[0,0,1]])
K_inv = np.linalg.inv(K)
#Testing: generate map_0 after first scan

MAP_0 = np.zeros([600, 600])
ranges0 =  lidar_ranges[:,0]
x0 = (0,0,0)

log_map_0 = mapping(MAP_0, ranges0,T_bl, x0)

MAP_0 = binary_map(log_map_0)

#particle filter prediction
MAP = log_map_0
N_par = 10
mu = np.zeros([3,N_par])
alpha = 1./N_par*np.ones(N_par)
x_t = np.zeros(3) #robot pose
N_thres = N_par/4
pose = []
trajectory = []

for t in range(len(tau)):#0,lidar_ranges.shape[1]):#
    FR,FL,RR,RL = encoder_counts[0,t],encoder_counts[1,t],encoder_counts[2,t],encoder_counts[3,t]
    V_R = (FR+RR)/2*0.0022/tau[t]
    V_L = (FL+RL)/2*0.0022/tau[t]
    V = (V_R+V_L)/2
    for i in range(N_par):
        x,y,theta = mu[0,i], mu[1,i], mu[2,i]
        
        x = x + tau[t]*V*np.sinc(omega[t]*tau[t]/2)*np.cos(theta+omega[t]*tau[t]/2) + normal(0,0.01)#vt
        y = y + tau[t]*V*np.sinc(omega[t]*tau[t]/2)*np.sin(theta+omega[t]*tau[t]/2) + normal(0,0.01)#wt
        theta = theta + omega[t]*tau[t]+normal(0,0.001)
        mu[0,i], mu[1,i], mu[2,i] = x,y,theta
        Neff = 1/np.sum(alpha**2)
        if Neff <= N_thres:
            mu,alpha = resample(mu, alpha)
    #update
    mu,alpha = update(mu,alpha,lidar[:,t+1],MAP,T_bl)        
    x_t = mu[:, alpha.argmax()]#use best particle to predict robot pose
    pose.append(x_t)
    m_ratio = 10
    x_pos = np.round((x_t[0]-10)*m_ratio+300).astype('int')
    y_pos = np.round(300-(x_t[1]-5)*m_ratio).astype('int')
    trajectory.append([x_pos,y_pos])
    #MAPPING
    #if t%10 == 0 or t == len(tau):
    MAP = mapping(MAP,lidar[:,t], T_bl, x_t)
    if t%100 == 0:
        print('current time step =',t)
        plot_map(MAP,trajectory)
MAP_0 = -binary_map(MAP)
trajectory = np.array(trajectory)  
#plot trajectory
fig = plt.figure(figsize=(10,10))
            
plt.imshow(MAP_0,cmap="gray") 
#plt.scatter(trajectory[:,0],trajectory[:,1],s=1,c='b')
plt.show()

#sychronizing time steps of robot pose, rgb img to disparity
rgb_steps = []
pose_sync =[]
for t in disp_stamps:
    diff =[]
    for i in range(len(rgb_stamps)):
        diff.append(np.absolute(t-rgb_stamps[i]))
    val,idx = min((val,idx) for (idx,val) in enumerate(diff))
    rgb_steps.append(idx)
    diff = []
    for j in range(len(encoder_stamps)-1):
        diff.append(np.absolute(t-encoder_stamps[j+1]))
    val,idx = min((val,idx) for (idx,val) in enumerate(diff))
    #lidar_range_sync[:,i+1] = lidar_ranges[:,idx]
    pose_sync.append(pose20[:,idx])
    
    
pose_sync = np.vstack(pose_sync).T   

disp = cv2.imread(os.path.join(Disparity, "disparity20_1.png"),-1)  
  
#texture mapping
uv = np.zeros((3,1))
for i in range(disp.shape[0]):
    for j in range(disp.shape[1]):
        m = np.array([j,i,1]).reshape(3,1)
        uv = np.column_stack((uv,m))
uv=np.delete(uv,0,axis =1)

MAP_0 = np.zeros([600, 600, 3 ])
tmap = MAP_0

for t in range(len(disp_stamps)):
    disp = cv2.imread(os.path.join(Disparity, "disparity20_%d.png"%(t+1)),-1)
    step = rgb_steps[t]
    rgb = cv2.imread(os.path.join(RGB, "RGB20_%d.png"%(step+1)),-1)
    rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    x_t = pose_sync[:,t]
    tmap = texture(uv,disp,rgb,x_t,tmap)

fig = plt.figure(figsize=(10,10))    
plt.imshow(tmap)
    
    
    
    
      