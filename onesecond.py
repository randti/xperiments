n=int(input())
a=list(map(int,input().split()))
n=int(input())
b=list(map(int,input().split()))
d=[]
for i in b:
  for j in a:
    i=i%j
  d.append(i)
print(*d)

