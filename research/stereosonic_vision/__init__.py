from flask import Blueprint, render_template

_stereosonic_vision = Blueprint('stereosonic_vision', __name__, template_folder='templates', static_folder='static', static_url_path='/static')
global cv

def set_cv(cv_path):
    global cv
    cv = cv_path

@_stereosonic_vision.route('/')
def home():
    return render_template('stereosonic_vision.html', cv=cv)

