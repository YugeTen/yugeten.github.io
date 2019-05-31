from flask import Blueprint, render_template

_visdial_cca = Blueprint('visdial_cca', __name__, template_folder='templates', static_folder='static', static_url_path='/static')
global cv

def set_cv(cv_path):
    global cv
    cv = cv_path

@_visdial_cca.route('/')
def home():
    return render_template('visdial_cca.html', cv=cv)

