def hir(a):
    a=a//60
    while a>=24:
        a=a-24
    return a
x,y=map(int,input().split())
a,b=map(int,input().split())
p,q=map(int,input().split())
d=x*60+y
d1=a*60+b
d2=p*60+q
d2=d2-d
if d2<0:
    d2=d2+24*60
d1=d1+2*d2
if d1//(60)>=24:
    print(hir(d1),d1%60)
else:
    print(d1//60,d1%60)