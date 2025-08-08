# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:25:28 2025

@author: human
"""

def gen():
    for i in range(5):
        yield i
        
my_gen = gen()
print(my_gen)