# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:53:12 2025

@author: human
"""

from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import pandas as pd



options = Options()
service = Service(
    executable_path=
    "E:/최남회/파이썬개발에대한파일모음/selenium/edgedriver_win64/msedgedriver.exe"
    )
driver = webdriver.Edge(service=service,options=options)
driver.get("https://www.google.com")
info_element = driver.find_element(By.CSS_SELECTOR,".MV3Tnb")
info_element.text

login_button = driver.find_element(By.CSS_SELECTOR,".gb_A")
login_button.click()

sub_titles = driver.find_elements(By.CSS_SELECTOR, ".link_nav")
[i.text for i in sub_titles]
sub_titles[0].click()
tb_element = driver.find_element(By.CSS_SELECTOR, "table")
target = tb_element.find_elements(By.CSS_SELECTOR, "tbody")[0]
table_html = tb_element.get_attribute("outerHTML")
df = pd.read_html(table_html)[0]

#드라이브 끄려고 할때 방법
driver.quit()