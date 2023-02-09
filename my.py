from flask import Flask,render_template,request
import pickle
import numpy as np
with open('mfile.pkl','rb') as model_file:
    model = pickle.load(model_file)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    user_data = np.zeros(15)
    user_data[0] = request.form['Hydrogen']
    user_data[1] = request.form['Oxigen'] 
    user_data[2] = request.form['Nitrogen']
    user_data[3] = request.form['Methane']
    user_data[4] = request.form['CO'] 
    user_data[5] = request.form['CO2']
    user_data[6] = request.form['Ethylene']
    user_data[7] = request.form['Ethane'] 
    user_data[8] = request.form['Acethylene']
    user_data[9] = request.form['DBDS'] 
    user_data[10] = request.form['Power factor']
    user_data[11] = request.form['Interfacial V']
    user_data[12] = request.form['Dielectric rigidity'] 
    user_data[13] = request.form['Water content']
    user_data[14] = request.form['Health index']
    
    

    
    # scaled_user_data= scaled_user_data.reshape(1,3)
    output = model.predict([user_data])
    print(output)
    result = output

   

    return render_template('index.html',customer_cluster=output[0])

if __name__ == '__main__':
    app.run(debug=True)
  