a,b,c=(map(int,input().split()))
p=[]
for i in range(1,a+1):
    p=i+1
    d=p+1
    if b>p and c>d:
        j=1
    else:
        print(i+p+d)
        break



