# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:49:33 2025

@author: human
"""

#이메일 qkdrk777777@naver.com

#메일명 " [python 기초] 최남회
# >.py 파일 첨부
# 향 후 회신 예정

#문제 내용은 기입하지 말 것
# 1.1)문제

input_number = int(input("참석자의 수를 입력하시오."))
chicken = input_number * 1
bear = input_number * 2
cake = input_number * 4

print(f"참석자의 수 : {input_number} \n, 치킨의 수 : {chicken} \n, 맥주의 수 : {bear} \n,케익의 수 : {cake}")

# 1.2)문제
x = 10
y = 20
temp = x
x = y
y = temp
print(f"x의 값 : {x}, y의 값 : {y}")

# 1.3)문제
#답 : 6번

# 1.4)문제

def cal(x) :
    y = 3*x**2 + 7*x + 9
    return y

x = 2
y = cal(x)
print (f"{x}의 값 : {y}")

# 1.5)문제
#연산자 우선 순위
# 답 : 4번() >> 1번** >> 2번*,/,//,% >> 3번 +-

# 1.6)문제

import math
def weight(r) :
    pi = math.pi
    wel = (4/3)*pi*r**3
    return wel

r = 3
wel = weight(r)
print(f"반지름의 길이가 {r}인 구의 부피 : {wel}")

# 1.7)문제
input_engWord = input("영어 단어 하나를 입력하시오 :")
remove_list = ["a","e","i","o","u"]
result = []

for i in input_engWord:
    if not i in remove_list :
        result.append(i)
        
print(result)
print("".join(result))

# 1.8)문제
a = "I'm happy"
print(a)

# 1.9)문제
food = ["milk","eggs","cheese","butter","cream"]
result = []

for i in food:
    result.append(i[0])
    
print(result)

# 1.10)문제
def rainNumber() : 
    rain = int(input("강수량을 입력하세요. : "))
    if rain > 50:
        print("집에서 안나가고 TV를 본다")
    elif rain > 30:
        print("차 끌고 카페에 간다")
    elif rain > 0:
        print("우산 들고 근처 공원에 간다")
    else:
        print("공원에 가서 산책을 한다.")
        
rainNumber()

# 1.11)문제
def weightNum() : 
    weight = int(input("수화물의 무게를 입력하세요.(Kg) :"))
    price = 0
    try :
        if weight >= 40 :
            raise ValueError()
        elif weight >= 20 :
            price = 20000
        else :
            price = 0
    except ValueError as ve :
        print(f"40킬로 무게가 발생하여 배송 불가합니다.")
    else :
        print (f"입력한 무게 : {weight} Kg \n , 지불해야 하는 금액 : {price} 원")
    finally :
        print(f"함수를 종료합니다.")
        
weightNum()

# 1.12)문제
class animal() :
    def __init__(self) :
       self.act = "동물은 움직인다."
        
    def live(self) :
        print(self.act)
        
class dog(animal) :
    def __init__(self) :
        self.act = "강아지는 4발로 움직인다."
        
    def live(self) :
        print(self.act)
        
class human(animal) :
    def __init__(self) :
        self.act = "인간은 2발로 움직인다."
        
    def live(self) :
        print(self.act)

animal = animal()
animal.live()       
dog = dog()
dog.live()
human = human()
human.live()
