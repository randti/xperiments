import requests
from bs4 import BeautifulSoup as bs
headers={'accept':'*/*',
         'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 YaBrowser/18.11.1.805 Yowser/2.5 Safari/537.36'}
base_url='https://hh.ru/search/vacancy?area=1&search_period=3&text=python&page=0'
def hh_parse(base_url,headers):
    jobs=[]
    session=requests.Session()
    request=session.get(base_url,headers=headers)
    if request.status_code==200:
        soup=bs(request.content,'html.parser')
        divs=soup.find_all('div',attrs={'data-qa':'vacancy-serp__vacancy'})
        divs = soup.find_all('div', attrs={'data-qa': 'vacancy-serp__vacancy'})
        for div in divs:
            title=div.find('a',attrs={'data-qa':'vacancy-serp__vacancy-title'}).text
            href= div.find('a', attrs={'data-qa': 'vacancy-serp__vacancy-title'})['href']
            company = div.find('a', attrs={'data-qa': 'vacancy-serp__vacancy-employer'}).text
            text1=div.find('div', attrs={'data-qa': 'vacancy-serp__vacancy_snippet_responsibility'}).text
            text2=div.find('div', attrs={'data-qa': 'vacancy-serp__vacancy_snippet_requirement'}).text
            content=text1+' '+text2
            jobs.append({
                'title':title,
                'href':href,
                'company':company,
                'content':content
            })
            print(jobs)
    else:
        print('Error')
hh_parse(base_url,headers)