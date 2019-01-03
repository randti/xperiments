"""
import requests
s=requests.get('https://m.news.yandex.ru')
print(s.text)

import requests
import html2text
s=requests.get('https://www..ru')
d = html2text.HTML2Text().handle(s.text)
print(d)
"""
"""
from urllib import request
import sys
myUrl="https://img3.goodfon.ru/original/1366x768/1/cf/gory-cvety-rassvet-solnce.jpg"
myFile="C:\\alo\\mk.jpg"
request.urlretrieve(myUrl,myFile)
"""