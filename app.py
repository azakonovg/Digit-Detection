from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    displayed_text = ''
    if request.method == 'POST':
        input_text = request.form.get('text', '')
        displayed_text = f"{input_text} {input_text}"
    return render_template('index.html', displayed_text=displayed_text)

if __name__ == '__main__':
    app.run(debug=True) 