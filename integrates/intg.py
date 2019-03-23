#ввести функцию в 20 строчке и нажать run
from numpy import  *
def pryam(f,a,b,n):
    h=float(b-a)/n
    result=abs(f(a+0.5*h))
    for i in range(1,n):
        result+=abs(f(a+0.5*h+i*h))
    result*=h
    return result
def trapez(f,a,b,n):
    h=float(b-a)/n
    result=0.5*((f(a))+(f(b)))
    for i in range(1,n):
        result+=abs(f(a+i*h))
    result*=h
    return result
# ВАША ФУНКЦИЯ,ЕСЛИ У ВАС ДВЕ ВВЕСТИ ИХ РАЗНОСТЬ
# НАПРИМЕР v=lambda x:sin(x)-x**2

v=lambda x:4-(x**2)
print('Кол-во разбиений, ввести и нажать enter')
#Чем больше тем лучше точность в районе 1000
n=int(input())
print('Отрезок, ввести два числа через пробел и нажать enter')
m,k=map(float,input().split())
print('Вычисленное значение методом прямоугольников:',pryam(v,m,k,n))
print('Вычисленное значение методом трапеции:',trapez(v,m,k,n))