"""
import random
a=[random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(10**4)]
a.append('.')
d={a:0 for a in 'abcdefghijklmnopqrstuvwxyz'}
i=0
while a[i]!='.':
    d[a[i]]+=1
    i+=1
d=sorted(d.items(),key=lambda d:d[1])
for i in d:
    if i[1]!=0:
        print(i[0],end="")
"""
"""
import webbrowser
webbrowser.open('http://google.co.kr', new=2)
"""

# from sympy import *
# x=Symbol('x')
# print(integrate(x,(x, 32, 7)))
a,b=map(int,input().split())
c=list(map(int,input().split()))
c=sorted(c)

k=0
i=0
j=a-1
d=-1
while i!=j:
    k+=1
    if c[i]*c[j]>b:
        j-=1
    elif c[i]*c[j]<b:
        if c[i]*c[j]>d:
            d=c[i]*c[j]
        i+=1
    else:

        print(d)
        break
print(d)
print(k)


