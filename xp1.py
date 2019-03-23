"""
n=int(input())
n=str(n)
p=0
while len(n)!=1:
    n=[int(i) for i in n]
    n=str(sum(n))
    p+=1
print(n,p,input())
"""
print(len(input()))