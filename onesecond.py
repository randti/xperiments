n=int(input())
a=list(map(int,input().split()))
a1=max(a)
b=set()
for i in range(len(a)):
  if a1%a[i]==0:
    b.add(a[i])
a=set(a)
if a-b:
  print(a1,max(a-b))
else:
  print(a1,a1)



