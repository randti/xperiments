import json
import os
import hashlib
blockchain_dir=os.curdir+'/blockchain/'
def get_hash(filename):
    file=open(blockchain_dir+filename,'rb').read()
    return hashlib.md5(file).hexdigest()
def get_files():
    files = (os.listdir(blockchain_dir))
    files = [int(i) for i in files]
    files = sorted(files)
    return files
def check_integrity():
    files=get_files()
    result=[]
    for file in files[1:]:
        f=open(blockchain_dir+str(file))
        h=json.load(f)['hash']
        prev_file=str(file-1)
        actual_hash=get_hash(prev_file)

        if h==actual_hash:
            result.append([True,prev_file])
        else:
            result.append([False,prev_file])
    return result
def write_block(name,amount,to_whom,prev_hash=''):
    files=get_files()
    prev_file=files[-1]
    filename=str(prev_file+1)
    prev_hash=get_hash(str(prev_file))
    # print(filename)
    data={'name': name,
          'amount':amount,
          'to_whom':to_whom,
          'hash':prev_hash}
    with open(blockchain_dir+filename,'w') as file:
        json.dump(data,file, indent=4,ensure_ascii=False)
def main():
    write_block(name='oleg',amount=5,to_whom='ksenya')
if __name__=='__main__':
    check_integrity()
    main()



