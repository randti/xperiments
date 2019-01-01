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