from flask import Flask, make_response, render_template
from functools import update_wrapper

app = Flask(__name__)


def nocache(f):
    def new_func(*args, **kwargs):
        resp = make_response(f(*args, **kwargs))
        resp.cache_control.no_cache = True
        return resp

    return update_wrapper(new_func, f)


@app.route("/")
@nocache
def run_eval():
    return render_template("webspeechdemo.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0')
