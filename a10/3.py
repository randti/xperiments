def viruslog(p):
    c=0
    for j in range(len(p)-1):
        for i in range(j,len(p)-1):
            a=p[j:i+1]
            a=a*3
            if p.count(a)!=0:
                c=1
                break
    if c==1:
        return True
    else:
        return False
p=input()
if p.count('bb')!=0 or viruslog(p)==True:
    print('NO')
else:
    print('YES')
