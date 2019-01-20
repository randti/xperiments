"""
import random
while True:
  a=random.randint(1,10000)
  a=str(a)
  a+='.txt'
  my_file = open(a, "w")
  my_file.write("Мне нравится Python!\nЭто классный язык!")
  my_file.close()
"""
from flask import Flask
from flask import render_template,redirect,url_for
from flask import request
from block import *
app=Flask(__name__)
@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        lender=request.form['lender']
        amount=request.form['amount']
        borrower=request.form['borrower']
        write_block(name=lender,amount=amount,to_whom=borrower)
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/checking')
def check():
    results=check_integrity()
    return render_template('index.html',results=results)

if __name__=='__main__':
    app.run(debug=True)

