c=list(map(int,input().split()))
a=list(map(int,input().split()))
b=list(map(int,input().split()))
a=set(a)
b=set(b)
b=list(b&a)
b=sorted(b)
print(*b)