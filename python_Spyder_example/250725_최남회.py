# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 09:11:29 2025

@author: human
"""

my_list = [1,2]
#dir:프로퍼티나 메소드 목록 출력
#help:는 함수의 설명 보고 싶을 때
dir(my_list) 

my_list = [1,2]
my_list.pop()
my_list

class Main:
    def __init__(self):
        self.type=1
    def fun(self):
        self.type = self.type + 1
        
class Sub(Main):
    def __init__(self):
        super().__init__()
        self.sub_type = 2
        
    def new_fun(self):
        return self.sub_type + self.type
    
sub_instance = Sub()
sub_instance.type
sub_instance.sub_type
sub_instance.new_fun()
str(sub_instance)
print(sub_instance)

def simple_generator(max):
    count = 1
    while count <= max:
        yield count
        count += 1

gen = simple_generator(30)
next(gen)

512 * 512 * 3 * 256 * 10000/8/1024/1024/1024 #234 GB
for i in gen:
    print(i)

