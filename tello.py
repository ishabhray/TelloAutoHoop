import numpy as np
import cv2 as cv
from djitellopy import Tello
import torch
from torch import nn
from math import *
import time

# tello=Tello()
# tello.connect()
# tello.streamon()
cap=cv.VideoCapture('data1.mp4')

r=10
theta=torch.linspace(0,2*pi,1000)
x=r*torch.cos(theta)
y=r*torch.sin(theta)
z=torch.zeros(1000)
zero = torch.tensor(0.)
one = torch.tensor(1.)
mtx = torch.tensor([[1.38207065e+03, 0.00000000e+00, 6.04678879e+02],[0.00000000e+00, 1.40234511e+03,3.47894185e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = torch.tensor([[3.84592539e-01, -2.16558956e+01, 5.49162969e-03, -8.14484970e-03, 2.19242479e+02]])

def get_circle(inp):
  img = np.zeros((1440, 1920),np.uint8)
  translation = np.array([[inp[0], inp[1], inp[2]]]).transpose()
  R1=np.array([[cos(inp[4]), 0, sin(inp[4])],
              [0, 1, 0],
              [-sin(inp[4]), 0, cos(inp[4])]])
  R2=np.array([[1, 0, 0],
              [0, cos(inp[3]), -sin(inp[3])],
              [0, sin(inp[3]), cos(inp[3])]])
  R=R1@R2
  obj=np.vstack((x,y,z))
  print(obj.shape)
  obj=R@obj+translation
  print(obj.shape)
  impt=mtx@obj
  s=impt[2,:]
  impt=impt/s
  print(impt.shape)
  for i in range(1000):
    img[(int)(impt[0][i])][(int)(impt[1][i])]=255
  return img

def get_ellipse_extreme(inp): #fucntion that gives you the extremas for a set of (x,y,z,alpha,beta)
  translation = inp[:3].reshape(3,1)
  R1=torch.vstack((torch.hstack((torch.cos(inp[4]), zero, torch.sin(inp[4]))), torch.hstack((zero, one, zero)), torch.hstack((-torch.sin(inp[4]), zero, torch.cos(inp[4])))))
  R2=torch.vstack((torch.hstack((one, zero, zero)), torch.hstack((zero, torch.cos(inp[3]), -torch.sin(inp[3]))), torch.hstack((zero, torch.sin(inp[3]), torch.cos(inp[3])))))
  R=R1@R2
  obj=torch.vstack((x,y,z))
  obj=R@obj
  obj=obj+translation
  img=mtx@obj
  s=img[2,:]
  img=img/s
  u_min_ind = torch.argmin(img[0, :])
  u_max_ind = torch.argmax(img[0, :])
  v_min_ind = torch.argmin(img[1, :])
  v_max_ind = torch.argmax(img[1, :])
  final = torch.vstack((img[:, u_min_ind], img[:, u_max_ind], img[:, v_min_ind], img[:, v_max_ind]))
  final = final[:, :2]
  return final

class Model(nn.Module):
  def __init__(self,x):
    super().__init__()
    self.x=nn.Parameter(x)
  def forward(self):
    return get_ellipse_extreme(self.x)

def training_loop(model,optimizer,goal,n=400):
  for _ in range(n):
    y=model()
    s=torch.sum(torch.pow(y-goal,2))
    s.backward()
    optimizer.step()
    optimizer.zero_grad()
  return s

def training_loop_early(model,optimizer,goal,n=400):
  loss=[]
  loss.append(100)
  cnt=0
  for _ in range(n):
    y=model()
    s=torch.norm(y-goal)
    s.backward()
    optimizer.step()
    optimizer.zero_grad()
    if abs(s-loss[-1])<5:
      cnt+=1
    else:
      cnt=0
    loss.append(s)
    if cnt==10:
      break
  return loss[-1]

cv.namedWindow('img',0)
# cv.namedWindow('imgReal',0)
# cv.namedWindow('img1',0)
# frame_read=tello.get_frame_read()
st=torch.tensor((0.0, 0.0, 20.0, 0.0, 0.0))
f=300
# cap=cv.VideoCapture(0)
while True:
  # frame=frame_read.frame
  _,frame=cap.read()
  gray=frame
  erosion_kernel=5
  erosion_it=1
  blur_kernel=5
  gray=cv.GaussianBlur(gray,(blur_kernel,blur_kernel),0)
  gray=cv.cvtColor(gray,cv.COLOR_BGR2YUV)
  light = (0, 125, 53)
  dark = (255, 200, 130)
  mask=cv.inRange(gray,light,dark)
  kernel = 2*np.ones((erosion_kernel,erosion_kernel),np.uint8)
  mask = cv.erode(mask,kernel,iterations = erosion_it)
  (numLabels, labels, stats, centroids)= cv.connectedComponentsWithStats(mask, 8, cv.CV_32S)
  mx=0
  idx=-1
  for i in range(1,numLabels):
    if stats[i, cv.CC_STAT_AREA]>mx:
      mx = stats[i, cv.CC_STAT_AREA]
      idx=i
  if idx==-1:
    print("loamded")
    continue
  mask=(labels == idx).astype("uint8") * 255
  mask=cv.dilate(mask,kernel,iterations = 2)
  mask=cv.erode(mask,kernel,iterations = 2)
  contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  if len(contours)!=2:
    print("Oops")
    continue
  if cv.contourArea(contours[0])>cv.contourArea(contours[1]):
    ellipse=contours[1]
  else:
    ellipse=contours[0]
  # cv.imshow('img1',mask)
  img=np.zeros(gray.shape,dtype=np.uint8)
  cv.drawContours(img, ellipse, -1, (0,255,0), 3)
  
  ellipse=np.array(ellipse)
  ellipse=np.squeeze(ellipse)
  u_min_ind,v_min_ind = np.argmin(ellipse,axis=0)
  u_max_ind,v_max_ind = np.argmax(ellipse,axis=0)
  goal= np.array((ellipse[u_min_ind],ellipse[u_max_ind],ellipse[v_min_ind],ellipse[v_max_ind]))
  for i in goal:
    cv.circle(img,(i[0],i[1]),10,(0,0,255),-1)
  # cv.imshow('img',img)
  goal=torch.tensor(goal)
  # start_time=time.time()
  m=Model(st)
  lr=0.1/600*torch.norm(get_ellipse_extreme(st)-goal)
  opt=torch.optim.Adam(m.parameters(),lr=lr)
  if f==30:
    f=1
    l=training_loop(m, opt, goal, n=400)
  else:
    f+=1
    l=training_loop_early(m, opt, goal, n=400)
  # print("Adam--> %s "%(time.time()-start_time))
  # cv.imshow('ell',get_circle(torch.tensor(m.x)))
  pred=get_ellipse_extreme(m.x)
  pred=pred.detach().numpy()
  for i in pred:
    cv.circle(img,(i[0],i[1]),10,(255,0,0),-1)
  img=cv.putText(img,str(l.item()),(0,479),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  cv.imshow('img',img)
  # cv.imshow('imgReal',img)
  k=cv.waitKey(1000//30) & 0xFF
  if k==27:
    break
cv.destroyAllWindows()