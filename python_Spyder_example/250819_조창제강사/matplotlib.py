# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:59:46 2025

@author: human
"""

# page 381
%matplotlib auto
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np 
data = np.arange(10)
data = np.random.RandomState(42).randn(10)
plt.plot(data)
data[2] = np.nan
plt.plot(data)

fig = plt.figure() # 빈 도화지 생성
ax1 = fig.add_subplot(2, 2, 1) # 2, 2 행렬 중 1번째
ax2 = fig.add_subplot(2, 2, 2) # 2, 2 행렬 중 2번째
ax3 = fig.add_subplot(2, 2, 3) # 2, 2 행렬 중 3번째
ax1.plot(data) # 2,2 행렬 중 1번째에 그림 그리기
ax1.cla() # 1번째에 그림 삭제하기
ax2.plot(data)

# 처음부터 하나의 그림을 쪼개서 생성도 가능
fig, (ax1, ax2) = plt.subplots(
    nrows=2, 
    gridspec_kw={"height_ratios":[2,7]}
    )
ax1.plot(data)



########################page 384
fig = plt.figure() # 빈 도화지 생성
ax1 = fig.add_subplot(2, 2, 1) # 2, 2 행렬 중 1번째
ax2 = fig.add_subplot(2, 2, 2) # 2, 2 행렬 중 2번째
ax3 = fig.add_subplot(2, 2, 3) # 2, 2 행렬 중 3번째
data = np.random.RandomState(42).randn(50).cumsum()
ax3.plot(data, color = "black", linestyle="dashed") 
ax3.cla()
ax3.plot(data, color = "black", 
         linestyle=":", linewidth = 3) 
ax1.cla()
# 히스토그램
ax1.hist(data, bins =10, color ="black",alpha=0.3)
# 산점도
data2 = np.random.RandomState(42).randn(50)
pts1 = np.arange(50)
pts2 = pts1 + data2
ax2.cla()
ax2.scatter(pts1, pts2, s=0.2)
fig.subplots_adjust(wspace=0, hspace=0) # 그래프 사이의 여백 제거

ax1.tick_params(
    left=False,bottom=False,
    labelbottom=False, labelleft=False)
ax1.tick_params(
    left=False,bottom=False,
    labelbottom=True, labelleft=True)

#ax1.set_xticks([],[])  # x 축 틱 & 레이블 삭제
#ax1.set_yticks([],[])  # y 축 틱 & 레이블 삭제

#ax2.set_xticks([],[])
#ax2.set_yticks([],[])

#ax3.set_xticks([],[])
#ax3.set_yticks([],[])
#ax2.tick_params(axis="x",bottom=False) # x 축 틱 삭제
#ax2.tick_params(axis="y",left=False)# y 축 틱 삭제
#ax2.set_xticklabels("")

##########page 389
fig = plt.figure()
ax = fig.add_subplot()
data2 = np.random.RandomState(42).randn(50).cumsum()
ax.plot(data2, color="black",linestyle="dashed", marker="o")

fig = plt.figure()
ax = fig.add_subplot()
ax.cla()
ax.plot(data2, color="black",linestyle="dashed")
ax.plot(data2, color="black",linestyle="dashed",
        drawstyle="steps-post", label="steps-post")
ax.legend()
ax.set_xticks([0, 25, 50])
ax.set_xticklabels(
    ["0m", "25m","50m"],
    rotation=30, fontsize=12)
ax.set_aspect("equal") #종횡비 1:1
plt.savefig("fig.png", dpi=600) #이미지 저장
"C:/Users/human/Desktop/" # 절대 경로
"./fig.png" # 상대 경로

%matplotlib auto
fig, ax = plt.subplots()
rect = plt.Rectangle(
    (0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color = "green", alpha= 0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
