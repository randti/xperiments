def others(a):
    l=0
    for i in a:
        if i!='*':
            l+=1
    return l
def ander(a,b):
    z=-1
    n=-1
    for i in range(len(a)):
        if n==-1 and a[i]=='?':
            n=i
        if a[i]!='*' and a[i]!='?':
            z=i
            break
    if z==-1:
        v=others(a)
        h=len(b)
        return h-v
    else:


        m=b.index(a[z])
        while True:
            if m+1==len(b):
                return m
            if b[m+1]==b[m]:
                m+=1
            else:
                return m



p=list(input())
s=list(input())
j=0
u=True
d=[]
for i in range(len(p)):
    if p[i]=='?':
        d.append(s[j])
        j+=1
    elif p[i]=='*':

        k=s[j:len(s)]
        x=others(p[i:len(p)])
        m=len(k)-x

        if m<=0:
            d.append([])
        elif x==0:
            if j==len(s)-1:
                d.append([])
            else:
                d.append(k)
                j+=len(k)-1
        else:
            m=ander(p[i:len(p)],k)
            d.append(k[0:m])

            j+=m
    elif p[i] == s[j]:
        d.append(s[j])
        j += 1
    else:
        print('NOT')
        print(d)
        u=False
        break
if u!=False:
    print('YES')
    print(d)
    print(123)

