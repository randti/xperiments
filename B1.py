def fact(c):
  if c==0:
    return 1
  else:
    d=1
    p=1
    while p<=c:
      d*=p
      p+=1
    return d
def sochitos(k,l):
  return fact(l)//(fact(l-k)*fact(k))
n,m,t=map(int,input().split())
a=4
d=0
while True:
  b=t-a
  if (a>=4 and a<=n) and (b>=1 and b<=m):
    d+=sochitos(a,n)*sochitos(b,m)
    a+=1
  elif a==n+1:
    break
  else:
    a+=1
print(d)