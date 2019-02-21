p=input()
s=input()
d=[]
c=[]
vis=[[False for i in range(10*5)] for j in range(10*5)]
def templat(i,j):
    try:
        if vis[i][j]:
            return False
        vis[i][j]=True
        if i==len(s) or j==len(p):
            if i == len(s) and j == len(p):
                return True

        elif p[j]=='?':
            if p[j-1]=='*':
                d.append('.')
            d.append(s[i])
            return templat(i+1,j+1)
        elif p[j]=='*':
            d.append('.')
            for k in range(i,len(s)+1):
                if templat(k,j+1):
                    return True
                else:
                    d.append(s[k])
        elif s[i]==p[j]:
            if p[j-1]=='*':
                d.append('.')
            d.append(s[i])
            return templat(i+1,j+1)
        else:
            return False
    except:
        if i == len(s) or j == len(p):
            if i == len(s) and j == len(p):
                return True
        elif p[j] == '?':
            if p[j - 1] == '*':
                d.append('.')
            d.append(s[i])
            return templat(i + 1, j + 1)
        elif p[j] == '*':
            d.append('.')
            for k in range(i, len(s) + 1):
                if templat(k, j + 1):
                    return True
                else:
                    d.append(s[k])

        elif s[i] == p[j]:
            if p[j - 1] == '*':
                d.append('.')
            d.append(s[i])
            return templat(i + 1, j + 1)
        else:
            return False
if templat(0,0):
    print('YES')
    print(d)
    c=1
    while len(d)!=0:
        k=d.pop(0)
        if k=='.':
            try:
                l=(d[0:d.index(k)])
                d=d[d.index(k)+1:len(d)]
            except:
                l=d
                d=[]
            if type(l) is list:
                print(*l,sep="")
        else:
            print(k)
else:
    print('NOT')


