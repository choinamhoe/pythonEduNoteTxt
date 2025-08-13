# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 09:02:01 2025

@author: human
"""

while True:
    커넥션
    
    데이터 주고 받는 과정
    
    
    
    
    
    
def Header():
    tag = ""
    return tag

def Footer():
    tag = ""
    return tag

def Home():
    tag = ""
    return tag

def About():
    tag = ""
    return tag

def Route(uri):
    header = Header()
    footer = Footer()
    
    route = {
        "/":Home(),
        "/about":About()
        }
    return header + route[uri] + footer

def App():
    tag = Route()
    return tag

