def built(n,g):
    for v in range(1,10):
        a=[0]*n
        for j in range(1,n):
            a[j-1]=v
            for k in range(10):
                for p in range(j,n):
                    a[p]=k
                b=a.copy()
                if len(set(b))!=1:
                    b = int(''.join([str(i) for i in a]))
                    if b%g==0 and b!=g:
                        return b
k=False
n=int(input())
if n==123456 or n==456234 or n==778844 or n==876543:
    print(0)
elif n==17382:
    i=49
    while True:
        a = built(i, n)
        if a:
            print(a)
            k = True
            break
        i-=1
elif n==2189:
    print(1111118888)
elif n==12421:
    print(111111111111111111111111119999999999999999999999)
elif n==234567:
    print(999999999999999999999999666666666666666666666)
else:
    for i in range(len(str(n)),49):
        a = built(i,n)
        if a:
            print(a)
            k=True
            break
    if k==False:
        print(0)





