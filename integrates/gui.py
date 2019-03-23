from sympy import *
from sympy.plotting import *
x=Symbol('x')
print('Functions')
a=input()
b=input()
print('values')
x1,x2=map(float,input().split())
p=a+'-'+'('+b+')'
p='sqrt'+'('+'('+p+')'+'*'+'('+p+')'+')'
c=abs(integrate(p,(x,float(x1),float(x2))))
# e=(integrate(abs(b), (x, float(d[0]), float(d[1]))))
print(c)
d=plot(a,b,(x,x1-2,x2+2))
d.save('1.png')
input()