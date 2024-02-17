from flask import *
import pandas as pd
import pickle
import re
import string
with open('ml_model.pkl', 'rb') as f:
    LR, DT, GBC, RFC, vectorization = pickle.load(f)
app=Flask(__name__)
@app.route("/")
@app.route('/res',methods=['GET','POST'])
def home():
    if request.method == "POST":
        def wordopt(text):
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub("\\W"," ",text) 
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)    
            return text
        def manual_testing(news):
            testing_news = {"text":[news]}
            new_def_test = pd.DataFrame(testing_news)
            new_def_test["text"] = new_def_test["text"].apply(wordopt) 
            new_x_test = new_def_test["text"]
            new_xv_test = vectorization.transform(new_x_test)
            pred_LR = LR.predict(new_xv_test)
            pred_DT = DT.predict(new_xv_test)
            pred_GBC = GBC.predict(new_xv_test)
            pred_RFC = RFC.predict(new_xv_test)
            # return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),output_lable(pred_DT[0]),output_lable(pred_GBC[0]),output_lable(pred_RFC[0])))
            return [pred_LR,pred_DT,pred_GBC,pred_RFC]
        news = request.form.get("newsName")
        manual_test=manual_testing(news)
        print(manual_test)
        output = max(manual_test,key=manual_test.count)
        # here output is taken by the max of the output by four methods [1,1,1,0] then output is 1(Not a fake News)
        print(output)
        return render_template("index.html",res=output[0])
    else:
        return render_template("index.html")
    
if __name__=='__main__':
    app.run(debug=True,port="5000")