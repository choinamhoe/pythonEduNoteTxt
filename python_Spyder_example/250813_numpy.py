# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 09:33:30 2025

@author: human
"""

# 자료형태가 1개이고 Array
# pip install numpy
import numpy as np # as는 너무 길어서 줄여서 사용하고 싶을 때 기재
#import numpy
#import pandas as pd
np.arange(1000000)
np.arange(1_000_000)
list(np.arange(2,10,2,np.int64))

# 시간 비교 (결론 넘파이가 빨라서 써야 한다)
my_arr = np.arange(100000)
my_list = list(range(100000))
%timeit my_arr2 = my_arr * 2
%timeit my_list2 = [i*2 for i in my_list]

#shape을 통해서 구조를 확인할 수 있다.
np.array([1,2,3]).shape
np.array([[1.5, -0.1, 3],[0,-3,6.5]]).shape
np.array([[1.5, -0.1, 3],[0,-3,6.5],[1,2,3]]).shape

np.array(
    [
     [
     [1.5, -0.1, 3],
     [0,-3,6.5]
     ],
     [
     [1.5, -0.1, 3],
     [0,-3,6.5]
     ]
     ]
    ).shape

data = np.array([[1.5, -0.1, 3],[0,-3,6.5]])
data * 10
data.shape
data.dtype
data+data
data+1

data = [6, 7.5, 8, 0, 1]
arr1 = np.array(data)
arr1
arr1.shape
arr1.dtype

data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
arr2

data3 = [1,2,3,4]
arr3 = np.array(data3)
arr2+arr3 # 마지막 shape가 같으면 연산이 일부 가능
arr1+arr2 # shape가 다르면 연산이 안됨

np.zeros((2,3)) #0값으로 행렬을 생성(2,3)
np.ones((2,3)) # 1값으로 행렬을 생성(2,3)
np.full((2,3),3) # 원하는 값으로 행렬을 생성(2,3)
# np.empty((2,3,2))
np.eye(4) # 정반행렬(4,4)

data4 = np.array([2.0,3.0,4.0])
data4.dtype
data5 = data4.astype(np.int64)
data5
data5.dtype
data4 = np.array([2.2,3.3,4.4])
data5 = data4.astype(np.int64)
data5

arr = np.array([[1,2,3],[4,5,6]])
arr
arr * arr
arr - arr
1/arr
arr **2
arr>2 #2보다 큰지 True False로 반환
arr[arr>2] # 2보다 큰 값을 추출
arr[arr>2]=6 # 2보다 큰 값을 6으로 할당
arr

# 브로드 캐스팅(복사한 값이 수정되었을 때 원본에 해당되는 값이 같이 변경되는 현상)
arr = np.arange(10)
arr[5]
arr[5:8]
arr

arr_slice = arr[5:8]
arr_slice[1] = 12345
arr_slice
arr

# 브로드 캐스팅 막으려면 copy() 함수로 복사해와서 사용해야 함
arr = np.arange(10)
arr[5]
arr[5:8]
arr

arr_slice = arr[5:8].copy()
arr_slice[1] = 12345
arr_slice
arr
""" 원본값이 변하지 않은 것을 확인 가능
array([0,1,2,3,4,12,12,12,8,9])
"""
arr2d = np.arange(1,10).reshape(3,3)
"""
array([
       [1,2,3],
       [4,5,6],
       [7,8,9]
       ])
"""
arr2d.shape
#2번째 행
arr2d[1]
arr2d[1,:]

#2번째행 2번째 값
arr2d[1][1]
arr2d[1,1]

#2번째 열의 값
arr2d[:,1]

#2번째 열의 2번째 값
arr2d[:,1][1]
arr2d[1,1]

arr3d = np.arange(1,13).reshape((2,2,3))
arr3d
arr3d[0]
arr3d[0].shape
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d

arr2d
arr2d[:2,1:]

names = np.array(["Bob","Joe","Will", "Bob","Will","Joe","Joe"])
data = np.array([[4,7],[0,2],[-5,6],[0,0],[1,2],[-12,-4],[3,4]])
names.shape, data.shape

data[names=="Bob"]
data[names=="Bob",0]
data[names=="Bob",1]

data[names!="Bob"]

data[(names=="Bob")|(names=="Will")]

#이름이 Joe 2번째 값을 8
data[names=="Joe"]
data[names=="Joe",1]
data[names=="Joe",1] = 8
data[names=="Joe"]

arr = np.zeros((8,4))
arr
for i in range(8):
    arr[i] = i

arr
arr[[4,3,0,6]]    
arr[[4,3,0,6],:]

arr[[-3,-1]]
arr[[-3,-1],:]

arr[[-3,-1],2:]
arr[[-3,-1],-2:]

np.arange(1,10).reshape((3,3)).T

arr = np.arange(32).reshape((8,4))
arr
# 홀수행에 짝수 열 선택해서 0으로 변경
arr[::2]
arr[::2,1::2] = 0

#변경된 값에서 (3,2)(5,2)를 8로 변경(파이썬이니가 (2,1),(4,1))
arr[[2,4],1] = 8
arr

#전치행렬 행과 열의 위치를 변경 T를 통해 활용
arr = np.arange(15).reshape((3,5))
arr
arr.T

#내적
# arr.T[0,:] * arr[:,0] = 0, 25, 100 => 더해서 내적에 [0,0] 값
# arr.T[0,:] * arr[:,1] = 0, 30, 110 => 더해서 내적에 [0,1] 값
np.dot(arr.T, arr)
arr.T@arr

#표준정규분포(평균 = 0 표준편차 = 1) 4*4 행렬로 16개 랜덤 생성
np.random.standard_normal(size=(4,4))


np.random.randn(16).reshape((4,4))
np.random.randn(16)
#랜덤 값을 고정하고 싶을 때 RandomState로 난수 번호 고정(Seed)
np.random.RandomState(42).randn(16)

arr = np.arange(10)
np.sqrt(arr)
arr**(1/2)
arr

np.exp(arr)
np.log(arr) #자연로그
np.log10(arr)
np.log2(arr)

arr = np.array([0.21,0.56,0.64])
np.ceil(arr) #올림
np.floor(arr) #버림
np.round(arr) #정수 반올림
np.round(arr,1) #소수 n째 자리가지 반올림

np.cos(90) 
#코사인 함수는 라디안 값을 입력받는데 저희가  알고 있는 90은 각도
#라디안 값으로 변경이 필요.np.deg2rad 활용해주거나 np.pi/180을 곱해주어야 함
np.cos(np.pi/180*90) #np.pi/180을 곱해줘야 된다.
np.cos(np.deg2rad(90))
np.sin(np.deg2rad(90))
np.tan(np.deg2rad(90))

points = np.arange(-5,5, 0.1 )
points.shape
xs, ys = np.meshgrid(points,points)
#conda activate py_312
#pip install matplotlib
import matplotlib.pyplot as plt
z = np.sqrt(xs**2 + ys**2)
%matplotlib auto
plt.imshow(z, cmap = plt.cm.gray,extent=[-5,5,-5,5])
plt.colorbar()
plt.close()

# plt.scatter(points,points, c=points, cmap="viridis")
# plt.colorbar()

xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
xarr = np.array([i/10+1 for i in range(1,6)])
yarr = np.array([i/10+2 for i in range(1,6)])
cond = np.array([True]*5)
cond[[1,-1]] = False
cond

[(x,y,c) for x,y,c in zip(xarr,yarr,cond)]

#조건식이 True를 할당하면 1 아니면 2를 반환
result = 1 if True else 2

if True:
    result = 1
else:
    result = 2
result

[(x if c else y) for x,y,c in zip(xarr,yarr,cond)]

np.where(cond)
test = [[True,False,True],[False,True,False]]
np.array(test)
np.where(test) #True/False값을 가진 행렬의 위치(인덱스)를 찾을 때

#True 값인 경우 xarr,False값인 경우 yarr 값을 반환하고 싶을 때
np.where(cond,xarr,yarr)

arr = np.random.RandomState(42).randn(16).reshape((4,4))
np.where(arr>0, 2, -2)
arr.mean() #평균
arr.std() #표준편차
arr.var() #분산
arr.var()**(1/2) #표준편차

arr = np.arange(1,10).reshape(3,3)
arr
arr.mean(axis=1) #행별로 평균
arr.mean(axis=0) #열별로 평균

arr
arr.cumsum(axis=0)
arr.cumsum(axis=1)

(arr>5).sum() #5보다 큰 원소의 갯수

# 5보다 큰 짝수의 갯수
((arr>5) & (arr%2 == 0)).sum() 

names = np.array(["Bob","Joe","Will", "Bob","Will","Joe","Joe"])
np.unique(names)
np.unique(names).shape
np.unique(names).shape[0]

arr = np.arange(1,10).reshape(3,3)
np.save("E:/최남회/python_Spyder_example/test.npy",arr)
arr = np.load("E:/최남회/python_Spyder_example/test.npy")

np.linalg.inv
np.linalg.qr

"""
TMI
주성분 분석 :
    PCA : 상관계수(COR) 행렬을 고유값 분해한 뒤 Scree plot 통해서
    차원을 축소하는 기술
    #행렬 축소로 고유값 분해나 특이값 분해(SVD)를 사용하는데
    연산이 효율적이라서 특이값 분해를 사용하기도 함
"""
position = 0
walk = [position]
nsteps = 1000
for _ in range(nsteps):
    #np.random.randint(2,size=1) : 0또는 1 랜덤하게 1개 나오게
    if np.random.randint(2,size=1):
        step = 1
    else:
        step = -1
    position = position + step
    walk.append(position)

plt.plot(walk[:100])        
plt.scatter(range(100),walk[:100])

nsteps = 1000
rng = np.random.default_rng(seed=12345)
draws = rng.integers(0,2, size=nsteps)
steps = np.where(draws == 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk)

nwalks = 5000
nsteps = 1000
draws = rng.integers(0, 2, size=(nwalks, nsteps))
steps = np.where(draws == 0, 1, -1)
steps.shape
walks = steps.cumsum(axis=1)
plt.plot(walks[:3,:].T) # 5000번 반복 중 앞에서 3개 값만 시각화

#이미지 n장을 불러와서 n, 512,512,3의 형태로 만들고자 할 때 Type1
img1 = np.zeros((512,512,3))
img2 = np.zeros((512,512,3))
img3 = np.zeros((512,512,3))
arrs = list()
for img in [img1,img2,img3]:
    arrs.append(img)
np.array(arrs).shape

#이미지 n장을 불러와서 n, 512,512,3의 형태로 만들고자 할 때 Typ2
img1.shape
img1[np.newaxis].shape
arrs = list()
for img in [img1,img2,img3]:
    arrs.append(img[np.newaxis])
np.concatenate(arrs).shape

#추후에 딥러닝에서 제너레이터와 같이 활용할 예정