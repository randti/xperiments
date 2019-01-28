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



def maxlogn(a):
  def resare(a):
    n=len(a)//2
    a1=(a[0:n])
    a2=(a[n:len(a)])
    if min(a1,a2)==a1:
      return [min(a1,a2),a2]
    return [min(a1,a2),a1]
  b=[]
  while len(resare(a)[0])!=1:
    c=resare(a)
    a=c[0]
    b.extend(c[1])
  c=resare(a)
  a=c[0]
  b.extend(c[1])
  return (a,b)
a=list(map(int,input().split()))

d=[]
while len(a)!=0:
  b=maxlogn(a)
  d.append(b[0][0])
  a=b[1]
  print(1)

print(d)

