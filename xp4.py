def riv(a):
    i=a.pop(0)
    b=[]
    b.append(i)
    return b
a=list(map(int,input().split()))
c=riv(a)
print(a)