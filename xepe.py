a,b=map(int,input().split())
c=[0]*b
c[0]=1
m=0
for i in range(1,b):
    c[i]=c[i-1]*2
for j in range(a-b):
    m=0
    for i in range(b):
        m+=c[i]
    for i in range(b-1):
        c[i]=c[i+1]
    c[b-1]=m
print(m)