def built(n):
    d=[]
    for v in range(1,10):
        a=[0]*n
        for j in range(1,n):
            a[j-1]=v
            for k in range(10):
                for p in range(j,n):
                    a[p]=k
                b=a.copy()
                if len(set(b))!=1:
                    b=int(''.join([str(i) for i in a]))
                    d.append(b)
    return d
n=int(input())
k=False
for i in range(len(str(n))+1,50):
    a=built(i)
    a=sorted(a)
    a = list(filter(lambda x: x % n == 0, a))
    if len(a)!=0:
        print(a[0])
        k=True
        break
if k==False:
    print(0)
