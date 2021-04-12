from flask import Flask, render_template, request
#  We import our trained model from the file where it's stored
from ml_model import model




app = Flask(__name__)

# now that I have created my flask app I have to define as many routes as I need. 


@app.route('/')


# beneath the route definition we need to define a function that will be called
# when that route gets a request

def main():
    return "Welcome to my Flask"

@app.route('/BiasDetector', methods = ['GET', 'POST'])



def BiasDetector():
    if request.method == "POST":
        if len(request.form['text']) != 0:
            text = request.form['text']
            # result = [4.]
            # feed it to the model
            result = model(text)
            print(result)
            return render_template('index.html', text = text, results = result)
        else:
             return render_template('index.html')

    else:
        return render_template('index.html')








# if this file is run directly, the following line of code is executed.
if __name__ == "__main__":
    app.run(debug=False)
    
    







