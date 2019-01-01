a=list(map(int,input()))
l=250
p=[0]*500
p[l]=1
for i in a:
    if i==1:
        l+=1
        p[l]=1
    else:
        l+=-1
        p[l]=1
print(p.count(1))