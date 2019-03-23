n,m=map(int,input().split())
a=[[0]*n for i in range(n)]
for i in range(m):
    b,k=map(int,input().split())
    a[b-1][k-1]=1
timer=[0]
tin=[0]*n
tout=[0]*n
used=[0]*n
p=[0]*n
cycle=[False]
def dfs(v):
    used[v]=1
    timer[0]+=1
    tin[v]=timer[0]
    for i in range(n):
        to=a[v][i]
        if to==1:
            if used[i]==0:
                p[i]=v
                dfs(i)
            elif used[i]==1 and i!=p[v]:
                cycle[0]=True
    timer[0]+=1
    tout[v]=timer[0]
    used[v]=2
dfs(0)
for i in range(len(used)):
    if used[i]==0:
        dfs(i)

if cycle[0]==False:
    d=[]
    for i in range(n):
        d.append([tin[i],tout[i],i])
    d=sorted(d,key=lambda x:x[1],reverse=True)
    for i in d:
        print(i[2]+1,end=" ")
else:
    print(-1)

