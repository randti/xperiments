step=0
k=int(input())

d=list(map(int,input().split()))

a=d.copy()
a=sorted(a)
p=[[-1,1]]
left=0
right=len(a)-1
while right!=left:
    step+=1
    if a[left]*a[right]>k:
        right-=1
    elif a[left]*a[right]<k:
        if a[left]*a[right]>(p[len(p)-1][0]*p[len(p)-1][1]):
            p.append([a[left],a[right]])
        left+=1
    else:
        p.append([a[left],a[right]])
        break

print(p[len(p)-1][0]*p[len(p)-1][1])
print(d.index(p[len(p)-1][0])+1,d.index(p[len(p)-1][1])+1)


# s=4
# a=[0]*s
# n=int(input())
# for i in range(s):
#     a[i]=int(input())
# mi=a[0]
# p=2001
# for i in range(s,n):
#     k=i%s
#     if a[k]<mi:
#         mi=a[k]
#     a1=int(input())
#     if a1+mi<p:
#         p=a1+mi
#     a[k]=a1
# print(p)

# a=[0]*10
# n=int(input())
# for i in range(n):
#     g=int(input())
#     g=str(g)
#     for j in g:
#         a[int(j)]+=1
# c=0
# d=0
# for i in range(len(a)):
#     if a[i]>c:
#         c=a[i]
# for i in range(len(a)-1,0,-1):
#     if a[i]==c:
#         print(i,end=" ")


