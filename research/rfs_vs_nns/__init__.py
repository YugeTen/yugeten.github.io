from flask import Blueprint 

_rfs_vs_nns = Blueprint('rfs_vs_nns', __name__, template_folder='templates', static_folder='static', static_url_path='/static')

