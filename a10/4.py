n=int(input())
a=[]
c=[]
d=[]
for i in range(n):
    a.append(int(input()))
    if a[-1]>100:
        d.append([a[-1],i])
    else:
        c.append([a[-1],i])


d=sorted(d)
c=sorted(c)[::-1]
l=[]
for i in c:
    x=0
    for j in d:
        if j[1]!='x' and i[1]>j[1]:
            l.append(i[1]+1)
            x=1
            i[1]='x'
            j[1]='x'
z=0
x=0
l=sorted(l)
for i in c:
    if i[1]!='x':
        z+=i[0]
    else:
        x+=1
u=0
for i in d:
    u+=i[0]
print(z+u)
print(len(d)-x,x)
for i in l:
    print(i)


