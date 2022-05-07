from flask import Flask, render_template, request
import crop_pred as cp

app = Flask(__name__)


@app.route("/", methods = ["GET","POST"])
def home():
    if request.method == "POST":
        inputData=[]
        temp = request.form
        for x in temp.values():
            inputData.append(int(x))
        pred = cp.crop_pred(inputData)
    else:
        pred=[""]
    return render_template("predict.html", pred_data = pred[0])

if __name__ == "__main__":
    app.run(debug=True)