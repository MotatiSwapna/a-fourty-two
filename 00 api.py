from flask import Flask

app = Flask('buffalo')

@app.route('/')
def f1():
    return "hello world"

@app.route('/me')
def f2():
    return "<input type='button'>Don't Click me</input>"

@app.route('/sum/<n1>/<n2>')
def f3(n1, n2):
    return n1+n2

app.run(host="0.0.0.0", port=3500)
