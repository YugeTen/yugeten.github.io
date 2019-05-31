from flask import Blueprint, request, render_template, url_for, redirect, send_from_directory
import os, uuid, pickle, itertools, requests, json
from werkzeug.utils import secure_filename

_flipdial = Blueprint('flipdial', __name__, template_folder='templates', static_folder='static', static_url_path='/static')

global params
global chat, img_to_load, current_img_path, current_cap

def set_global_params(cv_path, vd_host, vd_port):
    global params, chat
    params = {}
    params['maxchatlen'] = 10
    params['imglist'] = os.listdir(os.path.join(_flipdial.static_folder, 'thumbnails'))
    params['caps'] = pickle.load(open(os.path.join(_flipdial.static_folder, 'cap_dict'), 'rb'))
    params['cv'] = cv_path
    params['vd_host'] = vd_host
    params['vd_port'] = vd_port
    chat = []

def get_server_address():
    global params
    return 'http://' + params['vd_host'] + ':' + str(params['vd_port'])


def vd_answer_url(t, question, img_path, caption, history):
    global params
    if not caption: #blank string
	answer_method = '/get_answer_nocap?'
    else:
	answer_method = '/get_answer?'
    vd_url = get_server_address() + answer_method
    vd_url += 't=' + str(t)
    vd_url += '&question=' + question
    vd_url += '&img_path=' + img_path
    vd_url += '&caption=' + caption
    vd_url += '&history=' + json.dumps(history)
    return vd_url
    
def vd_check_ok():
    global params
    vd_url = 'http://' + params['vd_host'] + ':' + str(params['vd_port']) + '/'
    try:
        checkme = requests.get(vd_url)
    except requests.exceptions.RequestException as err:
        return False
    return True 

def ravel_answers(answers):
    return answers.split('**')

@_flipdial.route('/', methods=['GET', 'POST'])
def home():
    global params
    return render_template('flipdial.html', cv=params['cv'])

@_flipdial.route('/1vd_demo', methods=['GET', 'POST'])
def onevd_demo():
    global params
    global chat, img_to_load, current_img_path, current_cap
    default_img = 'COCO_val2014_000000524382.jpg'
    overscroll = 'column'

    if not vd_check_ok():
	current_img_path = os.path.join('images', default_img)
	img_to_load = ( url_for('flipdial.static', filename=os.path.join('images', default_img)), params['caps'][default_img])
    	return render_template('flipdial_nomodel.html', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])

    if request.method=='POST' and len(chat) < params['maxchatlen']:
	if 'msgboxheight' in request.form.keys():
            if float(request.form['msgboxheight'])+50 > 0:
	        overscroll = 'column-reverse'
        
        if 'reset' in request.form.keys():
            chat = []
            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], autofocus='autofocus', questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
            
        elif 'question' in request.form.keys():
            question = request.form['question']
            t = len(chat)*2

            # get answer by pinging server where flipdial model is loaded
            vd_url = vd_answer_url(t, question, current_img_path, current_cap, chat)
            answer = requests.get(vd_url) # get answer
            answers = ravel_answers(answer.text)

            item = (question, answers)
            chat.append(item)
            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], autofocus='autofocus', questiontext='', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
        
        elif 'img_to_load' in request.form.keys(): # select image from panel
            
            # empty chat
            chat = []
            
            # get image selected
            imgname = request.form['img_to_load']
            current_img_path = os.path.join('images', imgname)
            current_cap = params['caps'][imgname]
            img_to_load = ( get_preloaded_image_url(imgname), current_cap )

            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])

        elif 'upload_img' in request.files.keys(): # upload own image
            
            # empty chat
            chat = []

            # upload image to server and save with unique filename
            img_file = request.files['upload_img']
	    r = requests.post(get_server_address() + '/upload_image', files={'upload_img':img_file})
           
	    current_img_path = os.path.join('uploaded_images', r.text) 
	    current_cap = ''
            img_to_load = ( get_uploaded_image_url(r.text) , '' )

            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
    else: #standard page load

        # empty chat
        chat = []
        
        # set image and caption to default options
        current_img_path = os.path.join('images', default_img)
        current_cap = params['caps'][default_img]        
        img_to_load = ( get_preloaded_image_url(default_img), current_cap )

	return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['', overscroll], questiontext="enter your question...", imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'] )

@_flipdial.route('/active_dev', methods=['GET', 'POST'])
def active_dev():
    global params
    global chat, img_to_load, current_img_path, current_cap
    default_img = 'COCO_val2014_000000524382.jpg'
    overscroll = 'column'

    if not vd_check_ok():
	current_img_path = os.path.join('images', default_img)
	img_to_load = ( get_preloaded_image_url(default_img), params['caps'][default_img])
    	return render_template('flipdial_nomodel.html', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])

    if request.method=='POST' and len(chat) < params['maxchatlen']:
	if 'msgboxheight' in request.form.keys():
            if float(request.form['msgboxheight'])+50 > 0:
	        overscroll = 'column-reverse'
        
        if 'reset' in request.form.keys():
            chat = []
            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], autofocus='autofocus', questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
            
        elif 'question' in request.form.keys():
            question = request.form['question']
            t = len(chat)*2

            # get answer by pinging server where flipdial model is loaded
            vd_url = vd_answer_url(t, question, current_img_path, current_cap, chat)
            answer = requests.get(vd_url) # get answer
            answers = ravel_answers(answer.text)

            item = (question, answers)
            chat.append(item)
            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], autofocus='autofocus', questiontext='', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
        
        elif 'img_to_load' in request.form.keys(): # select image from panel
            
            # empty chat
            chat = []
            
            # get image selected
            imgname = request.form['img_to_load']
            current_img_path = os.path.join('images', imgname)
            current_cap = params['caps'][imgname]
            img_to_load = ( get_preloaded_image_url(imgname), current_cap )

            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])

        elif 'upload_img' in request.files.keys(): # upload own image
            
            # empty chat
            chat = []

            # upload image to server and save with unique filename
            img_file = request.files['upload_img']
	    r = requests.post(get_server_address() + '/upload_image', files={'upload_img':img_file})
           
	    current_img_path = os.path.join('uploaded_images', r.text) 
	    current_cap = ''
            img_to_load = ( get_uploaded_image_url(r.text) , '' )

            return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['demo', overscroll], questiontext='enter your question...', imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'])
    else: #standard page load

        # empty chat
        chat = []
        
        # set image and caption to default options
        current_img_path = os.path.join('images', default_img)
        current_cap = params['caps'][default_img]        
        img_to_load = ( get_preloaded_image_url(default_img), current_cap )

	return render_template('flipdial_1vd_demo.html', chat=chat, scroll=['', overscroll], questiontext="enter your question...", imgs=params['imglist'], img_to_load=img_to_load, cv=params['cv'] )

def get_uploaded_image_url(filename):
    image_url = get_server_address() + '/get_uploaded_image?'
    image_url += 'filename=' + filename
    return image_url

def get_preloaded_image_url(filename):
    global params
    image_url = get_server_address() + '/get_preloaded_image?'
    image_url += 'filename=' + filename
    return image_url

@_flipdial.route('/stem4britain_poster')
def stem4britain_poster():
    return send_from_directory(_flipdial.static_folder, 'flipdial_stem4britainposter.pdf')

@_flipdial.route('/2vd_demo')
def twovd_demo():
    return render_template('flipdial_2vd_demo.html')

@_flipdial.route('/paper')
def paper():
    return redirect('https://arxiv.org/abs/1802.03803')

@_flipdial.route('/code')
def code():
    return render_template('github.html')
