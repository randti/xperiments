"""
n=int(input())
a=list(map(int,input().split()))
b=[x for x in a if x>0]
b=sum(b)
p=a.index(min(a))
p1=a.index(max(a))
if p<p1:
    k=a[p+1:p1]
else:
    k=a[p1+1:p]
p=1
for i in k:
    p=p*i
print(b,p)
"""

"""
n=int(input())
b=[]
for i in range(n):
    a=input()

    if a.count('1')==a.count('2')==a.count('0'):
        if a[0:a.count('0')].count('1')!=0:
            b.append('NO')
        elif a[a.count('0'):a.count('0')+a.count('1')].count('2'):
            b.append('NO')
        else:
            b.append('YES')



    else:
        b.append('NO')
print('\n'.join(b))
"""
"""
k=1
f=1
a=int(input())
while k<a:
    k+=1
    f*=k
print(f)
"""
a=list(map(int,input().split()))
for i in range(len(a)):
    for j in range(len(a)):
        if a[i]<a[j]:
            k=a[i]
            a[i]=a[j]
            a[i],a[j]=a[j],a[i]

print(a)