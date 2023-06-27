from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import CustomData, Predictpipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicteddata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Age=request.form.get('Age'),
            Sex=request.form.get('Sex'),
            chest_pain_type=request.form.get('chest_pain_type'),
            BP=request.form.get('BP'),
            Cholesterol=request.form.get('Cholesterol'),
            FBS_over_120=request.form.get('FBS_over_120'),
            EKG_results=request.form.get('EKG_results'),
            Max_HR=request.form.get('Max_HR'),
            Exercise_angina=request.form.get('Exercise_angina'),
            ST_depression=float(request.form.get('ST_depression')),
            Slope_of_ST=request.form.get('Slope_of_ST'),
            Number_of_vessels_fluro=request.form.get('Number_of_vessels_fluro'),
            Thallium=request.form.get('Thallium'),
        )

        pred_df=data.get_data_as_frame()
        print(pred_df)

        predict_pipeline=Predictpipeline()

        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")