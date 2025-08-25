# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 09:15:44 2025

@author: human
"""

import numpy as np

"""
np.arange(start, end-1, 간격 - 실수 가능) #range와 유사한 함수.실수간격으로도 줄수 있음.range는 정수형태만
np.array - (리스트나 튜플)을 넘파이 객체로 변환
"""
np.arange(2,10,2)
np.arange(2,10, 0.2)

my_list = [1,3,2]
my_arr = np.array(my_list)
my_arr
my_arr.shape[0]
my_arr.dtype
"""
np.array : 같은 구조에 어레이 여려개가 들어가 있는 리스트를 
           하나의 어래이로 만들 때에도 활용
np.concatenate
"""
#파일 경로를 4개 불러와서 이미지로 읽어들인 다음
#imgs 리스트에 추가했다고 가정
imgs = list()
for i in np.arange(6):
    img=np.zeros((512,512,3))
    imgs.append(img)
np.array(imgs)
"""
조창제 강사 코딩
"""
imgs = list()
for i in range(6):
    img=np.zeros((512,512,3))
    imgs.append(img)
[i.shape for i in imgs]
np.array(imgs).shape

imgs = list()
for i in np.arange(6):
    img=np.zeros((512,512,3))
    imgs.append(img[np.newaxis])
[i.shape for i in imgs]    
np.concatenate(imgs).shape
"""
조창제 강사 코딩
"""
imgs = list()
for i in range(6):
    img=np.zeros((512,512,3))
    imgs.append(img[np.newaxis])
[i.shape for i in imgs]    
np.concatenate(imgs).shape
np.concatenate(imgs)

# np.newaxis 는 차원을 추가
# 인덱싱 하듯이 사용이 가능
# 맨 마지막에 넣을 때는 ..., np.newaxis 이런식으로 기재가능
img[np.newaxis,:,:,:].shape 
img[:,np.newaxis,:,:].shape
img[:,:,np.newaxis,:].shape
img[:,:,:,np.newaxis].shape
img[...,np.newaxis].shape

"""
np.zeros,np.ones, np.full
np.random.randn,np.random.RandomState(n).randn
특정 구조의 특정 값을 가지는 행렬 생성 가능
"""
my_shape = (512,512,3)
np.zeros(my_shape)
np.ones(my_shape)
np.full(my_shape,3) #3으로 된 my_shape 구조의 행렬
np.random.randn(512*512*3).reshape(my_shape)
np.random.RandomState(42).randn(512*512*3).reshape(my_shape)

"""
arr.astype : 넘파이 객체의 타입을 변경
arr.shape : 넘파이 객체의 구조를 확인
arr.reshape : 넘파이 객체의 구조를 변경
"""
my_list = [1,3,2,2]
arr = np.array(my_list)
arr.shape
arr.dtype
arr1 = arr.astype(np.float32)
arr1
arr1.dtype
arr1.reshape((2,2))
arr1 = arr1.reshape((2,2))
arr1
"""
np.ceil : 올림
np.floor : 내림
np.round : 반올림
"""
np.ceil(3.2)
np.ceil(3.7)
np.ceil(-3.2)
np.ceil(-3.7)
np.floor(3.2)
np.floor(3.7)
np.floor(-3.2)
np.floor(-3.7)
np.round(3.2)
np.round(3.7)
np.round(-3.2)
np.round(-3.7)
np.round(3.22,1)
np.round(3.27,1)
np.round(-3.22,1)
np.round(-3.27,1)
num1 = 3.2
num2 = 3.7
f"{num1:0.1f}"
f"{num2:0.1f}"
f"{-num1:0.0f}"
f"{-num2:0.0f}"
np.floor(-3.2)
np.floor(-3.7)


"""
np.exp : 지수함수
np.log : 자연로그(밑이 e인 로그를 의미)
np.log10 : 상용로그(밑이 10인 로그를 의미)
np.log2 : 밑이 2인 로그
np.cos : 삼각 함수 중 cos
np.deg2rad : 각도를 라디안 값으로 변환하는 함수
* 로그 : 밑을 K번 거듭제곱해서 N이 나올수 있는 K
"""
np.exp(4)
np.log10(10)
np.log2(2)
np.cos(90)
np.cos(np.deg2rad(90))
"""
arr.mean, arr.std,arr.var,arr.sum,arr.cumsum
- axis = 0, 1
"""
arr = np.array([[1,3,2],[4,5,6]])
arr.mean()
arr.mean(axis=0) # 열 기준
arr.mean(axis=1) # 행 기준
#분신
arr.var()
arr.var(axis=0) # 열 기준
arr.var(axis=1) # 행 기준
arr.std()
arr.var()**(1/2)
#합계
arr.sum()
arr.sum(axis=0) # 열 기준
arr.sum(axis=1) # 행 기준
arr.cumsum()
arr.cumsum(axis=0) # 열 기준
arr.cumsum(axis=1) # 행 기준

"""
np.where : True 값의 위치를 찾아주는 함수
np.where(논리배열, True값 반환값,False 값 반환값)

if True:
    var1 = 1
else:
    var1 = 2
위 조건문을 한줄로 표현한 식
var1 = 1 if True else 2
"""
my_list = [True,False, True,True,False]
np.where(my_list)
np.where(my_list,1,2)
my_list = [
    [True,False,False],
    [True,True,False]]
np.where(my_list)
np.where(my_list, 1, 2)
[i for i in range(10)]
[i for i in np.arange(10)]
result = ["짝수" if i%2 == 0 else "홀수" for i in range(10)]
result
result = ["짝수" if i%2 == 0 else "홀수" for i in np.arange(10)]
result

result = []
for i in range(10):
    if i%2 == 0:
        result.append("짝수")
    else:
        result.append("홀수")
result

result = []
for i in np.arange(10):
    if i%2 == 0:
        result.append("짝수")
    else:
        result.append("홀수")
result

"""
브로드 캐스팅 : 데이터 중 일부를 새로운 변수에 할당 했을 대
새로운 변수의 값을 변환했음에도 불구하고 원본 데이터에 값이 변경되는 현상
"""
arr1 = np.arange(10)
arr1
arr2 = arr1[2:5] # 얕은 복사 : 구조만 복사해오고 값은 원본 값을 참조
arr2
arr2[2:] = -1
arr2
arr1 # 5번째 값이 -1로 변경

arr1 = np.arange(10)
arr2 = arr1[2:5].copy() # 깊은 복사
arr2[2:] = -1
arr2
arr1 # 원본값이 변경 되지 않음.


"""
잘 사용하지 않을 것 같은 함수
np.eye(n) nxn 정방 단위 행렬 생성
np.meshgrid : 교과서 163페이지 참조(파이썬 라이브러리를 활용한 데이터 분석 3판)
np.linalg.inv : 역행렬
np.linalg.qr : 행렬 분해 방식 중 하나 QR분해

단위 행렬  = 주 대각 성분이 1이고 나머지 성분이 0인 행렬
"""
np.eye(4)

