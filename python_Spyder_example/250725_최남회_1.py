# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:12:25 2025

@author: human
"""

for x in range(5):
    print(x)
    
for abcde in range(5):
    print(abcde)
    
for name in ["철수","영희","길동"]:
#    print(f"안녕! " + name)
    print(f"안녕! {name} ")
    
for x in range(10):
    #print(x,end=" ")
    print(x)
    
for c in "abcdef":
    #print(c,end=" ")
    print(c)

"""
7) 영어 단어 하나를 입력 받아,
    모음을 제거하고
    자음목록을 출력하시오(중복은 무시)
"""    
input_word = list(input("영어단어를 입력하세요."))
remove_list = {"a","e","i","o","u"}
result = set(input_word) - remove_list
result = list(result)

print(f"입력한 단어 : {input_word}, 자음만 출력 : {result}")

"""
7-1) 영어 단어 하나를 입력 받아,
    모음을 제거하고
    자음목록을 출력하시오(중복은 무시)
"""    
input_word = list(input("영어단어를 입력하세요."))
remove_list = {"a","e","i","o","u"}
remove_list = list(remove_list)
result = []
for i in input_word:
    # 모음에 포함되지 않으면
    if not i in remove_list:
        result.append(i)
print(f"""
      입력한 단어 : {input_word}
      , 자음만 출력 : {result}
      """)

"""
8) 1부터 사용자가 입력한 수 n까지 더해서 (1+2+3+...n)까지 
    계산하는 프로그램 작성
""" 
number = int(input("어디까지 계산할까요? :"))
result = 0

for i in range(number):
    result += i + 1
    
print(f"1부터 {number}까지의 정수의 합 = {result}")  

  

"""
9) 팩토리얼 계산.1부터 사용자가 입력한 수 n까지 곱한다 (1*2*3*...n)까지 
    계산하는 프로그램 작성
"""     
number = int(input("팩토리얼 어디까지 계산할까요? :"))
result = 1
for i in range(1,number+1):
    result *= i
    
print(f"{number}!은 {result}이다") 

"""
2단으로 9단까지 구구단 출력
""" 
for i in range(2,10):
    for j in range(1,10):
        print(f"{i} * {j} = {i * j}")

"""
10) 1부터 100사이의 모든 3의 배수의 합을 계산하여 출력하는 프로그램
    
"""  
result = 0
number_1 = int(input("3의 배수의 마지막 숫자를 입력하세요.:"))
for i in range(number_1):
    if i%3 == 0:
        result += i
        
print(f"입력된 숫자 : {number_1}, 3의 배수의 합 : {result}")

"""
11)숫자 4자리를 인풋으로 각자리수의 합을 구하는 프로그램
    예) 1234를 입력받으면 (1+2+3+4)를 출력하는 값을 계산
"""
input_number = input("숫자를 입력하세요.:")
result = 0
for i in list(input_number):
    result += int(i)
    
print(f"입력된 숫자 : {input_number},입력된 숫자의 각자리의 합 : {result}")

input_number = input("숫자를 입력하세요.:")
result = 0
while int(input_number) > 0 :
    digit = int(input_number) % 10
    result = result + digit
    input_number = int(input_number) // 10
print(f"입력된 숫자의 각자리의 합 : {result}")

"""
12)문자열을 조사아여서 알파벳 문자의 갯수,숫자의 갯수,스페이스의 갯수를 출력하는 프로그램 작성
"""
input_str = input("문자열을 입력하시오")
alphabat = list("abcdefghijklmnopqrstuvwxyz")
number = list("123456789")
space = " "

alphabat_count = 0
number_count = 0
space_count = 0
for i in input_str:
    if i in alphabat:
        alphabat_count += 1
    if i in number:
        number_count += 1
    if i in space:
        space_count += 1
        
print(f"""입력된 문자 : {input_str}
      , 알파벳 갯수 : {alphabat_count}
      , 숫자 갯수 : {number_count}
      , 공백 갯수 : {space_count}
      """)


"""
13)계좌번호를 입력하시오 -을 없애는 로직 추가
"""
input_str = list(input("계좌번호을 입력하시오 : "))
list1 = []
for i in input_str:
    if i !='-':
        list1.append(i)

result = "".join(list1)

print(f"입력된 계좌번호 : {input_str}, -제거한 결과 : {result}")
"""
주민등록번호 검증번호는 다음과 같습니다.
주민등록번호 앞자리 12자리가지의
값을 ABCDEF-GHIJKL 이라고 할때에
다음과 같이 구합니다.
나눈값의 나머지를 MOD라 하겠습니다.

검증번호 =
(11 - (2A+3B+4C+5D+6E+7F+8G+9H+2I+3J+4K+5L)MOD(11))
"""


def say_hello(name):
    print("안녕,"+ name)

say_hello("철수")
    
def say_hello(name,msg):
    print("안녕,"+ name + "야,"+ msg)

say_hello("영희","반가워")


def exprNumber(input_number) :
    result = int(input_number) ** 2
    print(f""" 
          입력받은 숫자 : {input_number}
          , 제곱근 : {result}
          """)
          
exprNumber(50)

result = 0
def maxNumber1(num1,num2) :
    if int(num1) > int(num2) :
        result = num1
        print(f"둘중에 큰 수 : {result}")
    elif int(num2) > int(num1) :
        result = num2
        print(f"둘중에 큰 수 : {result}")
    else :
        result = num2
        print(f"둘의 숫자는 같다 : {result}")
    
maxNumber1(40,40)  


def happyBirthday(name) :
    str = "생일축하합니다."
    print(f"{str}\n{str}\n사랑하는 {name} {str}")
          
happyBirthday("최남회")

"""
구의 부피를 계산하세요.
반지름이 r인구의 부피는 10인 경우 4188.790204786391
"""

def sphereVolume(r):
    pi = math.pi
    result = (4/3)*pi*r**3
    print(f"반지름 : {r},구의 부피 : {result}")
    
sphereVolume(10)

import math
math.pi
"""
default값이 있는 경우 무조건 뒤로 빼줘야 오류가 안남
"""

def greed(name, msg = "aaa"):
    print(name,msg)
    
#def greed(name, msg = "aaa",aaa):
#    print(name,msg)
    
#def greed(name, msg = "aaa",aaa, bbb = "aaa"):
#    print(name,msg)
    
def greed(name,aaa , msg = "aaa", bbb = "aaa"):
    print(name,msg)
    
"""
사칙연산을 수행하는 4개의 함수(add(),sub(),mul(),div())
10+20-30을 계산
"""    
def add(num1,num2):
    return num1+num2

def sub(num1,num2):
    return num1-num2

def mul(num1,num2):
    return num1*num2

def div(num1,num2):
    return num1/num2

num1= 10
num2 = 20
num3 = 30
print(f"{num1}+{num2}-{num3} = {sub(add(num1,num2), num3)}")
print(f"{num1}+{num2}*{num3} = {add(num1,mul(num2,num3))}")
class Calculator:
        
    def add(num1,num2):
        return num1+num2

    def sub(num1,num2):
        return num1-num2

    def mul(num1,num2):
        return num1*num2

    def div(num1,num2):
        return num1/num2
    
calc = Calculator()
add1 = calc.add(num1,num2)
result = calc.sub(add1,num3)
print(f"{num1}+{num2}-{num3} = {result} ")

a=0
def fun():
    print(a)
    
fun()

def fun():
    global a
    a = a + 1
    
fun()

def sub(mylist):
    mylist = [1, 2, 3, 4]
    print(mylist)
    
mylist = [10,20,30,40]
sub(mylist)

phone = "010-4149-9177"
def split_phone_number(phone):
    a,b,c = phone.split("-")
    return a,b,c

a,b,c = split_phone_number(phone)
a
b
c

result = split_phone_number(phone)
result

scores = []
for i in range(10):
    scores.append(int(input("성적을 입력하세요.")))
print(scores)

scores = [ 32, 56, 64, 72, 12, 37, 98, 77, 59, 69]
scores[0] = 80
scores

scores[1] = scores[0]
scores

i=4
scores[i] = 10
scores[i+2] = 20
scores

list1 = [12,"dog",180.14]
list2 = [
    list1
    ,(12,"dog")
    ,{1,2,5}
    ,{"4":32, 4:12}
    ]

{4,"4"}
list2
list2[0][2] # = 1 이런식으로 값 변경 가능

#3 값 입력 평균
my_list = []
count = 0
while True:
    count = count + 1
    if count==4:
        print(sum(my_list)/3)
        break
    input_text = input("성적을 입력하세요.")
    my_list.append(input_text)


my_list = []
for i in range(3):
    input_text = int(input("성적을 입력하세요."))
    my_list.append(input_text)
print(sum(my_list)/3)

my_list

my_list = [82, 43, 72]
result = []
for i in my_list:
    if i>80:
        result.append(i)
        
result

[i for i in my_list if i>80]

my_list
len(my_list)

#강아지 3마리 이름 입력하면
#a,b,c 형태로 출력되게 : 

result = ""
for i in range(3):
    input_dog_text = input("강아지의 이름을 입력하세요.")
    result += input_dog_text + ","
    #result = result + input_dog_text


print("강아지의 이름 ", result[:-1])

result = []
for i in range(3):
    input_dog_text = input("강아지의 이름을 입력하세요.")
    result.append(input_dog_text)
resultStr=", ".join(result)    
print("강아지의 이름 ", resultStr)

phone  = ["010","4214","7733"]
phoneStr = "-".join(phone)
print("폰번호 ", phoneStr)

shopping_list = ["두부","양배추","딸기","사과","토마토"]

for i in range(len(shopping_list)) :
    print("쇼핑리스트 목록 : " + shopping_list[i])

"""
쇼핑리스트의 첫글자만 가져오는 방법
"""  
    
result = []
for i in shopping_list:
    result.append(i[0])
    
result
sorted(result) #화면은 바뀌지만 원본은 그대로
result
result.sort() #원본자체를 바꿈
result

def func2(list):
    list[0] = 99
    
values = [0,1,1,2,3,5,8]
print(values)
func2(values)
print(values)

#range 1~10 짝수수만
#if문 써서
#리스트 컨프리헨션 해보기
[i for i in range(1,11) if i%2==0]

result = []
for i in range(1,11):
    if i%2==0 :
        result.append(i)

result

"""
파타고라스의 정리를 프로그램
a2+b2+c2를 한변의 길이가 30이하인 삼각형을 모두 찾는 프로그램
"""  

sum_length=100
tri_list=list()
for a in range(1,sum_length):
    for b in range(a+1,sum_length):
        if a+b<100:
            for c in range(b+1,sum_length):
                if a+b+c<sum_length:
                    sort_num=sorted([a,b,c])
                    #삼각 부등식(두변의 합이 다른 한변 보다 커야 삼각형이 생김)
                    if(sum(sort_num[:2])>sort_num[-1])&(a**2+b**2==c**2):
                        tri_list.append([a,b,c])
                        
tri_list


result = []
for i in range(1,7):
    result_b = []
    for j in range(1,7):
       result_b.append(i+j)
    result.append(result_b)  
    
result
################
# tuple
t1 = (1, 2, 3, 4, 5)
t1 = tuple([1,2,3,4,5])
t1[0] = 9
a = {3,7,4}
a
for i in a:
    print(i)
    
A = {1,2,3}
B = {3,4,5}
A|B # 합집합
A&B # 교집합
A-B # 차집합
dir(A)

class MyClass:
    def __init__(self):
        self.__mykey = "asdb"
        
    def get_my_key(self):
        return self.__mykey
    
    def __str__(self):
        print("테스트용")
    
my_class = MyClass()
my_class.__mykey
my_class.get_my_key()
my_class.__str__()

my_dict = {}
for key, value in zip(["a","b","c"],[1,3,[5,42,7]]):
    my_dict.update({key, value})
