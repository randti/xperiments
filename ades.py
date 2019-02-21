n=int(input())
a=list(map(int,input().split()))
b=[]
k=0
d=0
stack=[]
for i in range(len(a)):
    if a[i]==0:
        stack.append(0)
    d+=1
    if i==len(a)-1:
        b.append(d)
        break
    if len(stack)==2 and a[i+1]==0:
        b.append(d)
        d=0
        stack=[]
print(len(b))
print(*b)







