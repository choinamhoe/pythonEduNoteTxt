"""
1.문제
1)참석자 수를 맞추어서 
치킨(1인당 1마리)
,맥주(1인당 2캔)
,케익(1인당 4개)
을 출력하는 프로그램 작성
"""  

input_num = int(input("참석자의 수를 입력하시오:"))
chicken = input_num * 1
bear = input_num * 2
cake = input_num * 4
print(f"참석자의 수를 입력하시오:{input_num}\n , 치킨의 수 : {chicken} \n , 맥주의 수 :{bear} \n 케익의 수:{cake}")

"""
2.문제
1)변수x와 변수 y의 값을 서로 바꾸는 프로그램 작성
x = 10, y=20을 x=20, y=10으로 맞교환
"""  
x = 10
y = 20
temp = x
x = y
y = temp
print(f"x = {x},y = {y}")

"""
3.문제
3)다음중 변수가 될수 있는 것은
6번 a_2
"""  

"""
4.문제
4)y = 3x2+7x+9을 함수로 만들어보고 x=2일때 값을 출력하시오
""" 
def cal(x) :
    y = 3*x**2 + 7*x+9
    print("결과값 : "+ str(y))
    
cal(2)

"""
4.문제
5)연산자 우선순위
답 : 1등 ()
    2등 **
    3등 *,/,//,%
    4등 +,-
""" 
"""
4.문제
6)반지름이 r인 구의 부피를 구하는 함수를 만들고 ,r이 3일때 값을 출력하시오
""" 
import math
result = 0
def weight(r):
    pi = math.pi
    result = (4/3)*pi*r**3 
    print(f"입력된 반지름의 길이 : {r}\n , 구의 부피 : {result}")
    
weight(3)
"""
4.문제
7)영어 단어 하나를 입력 받아, 모음을 제거하고 자음 목록을 출력하시오(중복은 무시)
""" 
input_English = input("영어 단어를 입력하시오.:")
remove_list = ["a","e","i","o","u"]
result = []
for i in input_English:
    if not i in remove_list:
        result.append(i)

print(result)        
print("".join(result))

"""
4.문제
8)I'm happy
""" 
a = "I'm happy"
print(a)

"""
4.문제
9)["milk","eggs","cheese","butter","cream"]의 맨 앞 글자만 추출
    화면에 ["m","e","c","b","c"]를 출력하라는 뜻
""" 
list1 = ["milk","eggs","cheese","butter","cream"]
result1 = []
for i in list1:
    result1.append(list1[0])
    
result1
