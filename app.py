from flask import Flask, render_template, request
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from tensorflow.keras.models import load_model
# from joblib import dump, load
import pickle

app = Flask(__name__)

# Function to generate MIP dataset
# def generate_mip_dataset(num_problems, num_variables, num_constraints, coefficient_range):
    # Your function code here

# Function to train DNN model


# Function to predict initial variables
def predict_initial_variables(model, objective_coefficients):
  predicted_variables = model.predict(np.array([objective_coefficients]))[0]
  predicted_variables = np.where(predicted_variables < 0, 0, predicted_variables)
  return predicted_variables  

# Function to solve MIP with branch and bound


# Load pre-trained DNN model
# dnn_model = load_model('your_model.h5')
# Load the model back
with open('dnn_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    # Get user input from form
    num_variables = int(request.form['num_variables'])
    num_constraints = int(request.form['num_constraints'])
    return render_template('solve.html', variable_options=num_variables, num_constraints=num_constraints)
@app.route('/result', methods=['POST'])
def result():
    num_variables = int(request.form['num_variables'])
    num_constraints = int(request.form['num_constraints'])

    # Create variables
    variables = [LpVariable(f'x_{i}', lowBound=0, cat='Binary') for i in range(1, num_variables + 1)]
    # for i, var in enumerate(variables):
    #  var.setInitialValue(predicted_variables[i])

    # Create MIP problem
    mip_problem = LpProblem("BinaryMIPProblem", LpMaximize)

    # Add objective function
    objective_coefficients = [int(request.form[f'objective_{i+1}']) for i in range(num_variables)]
    initial=predict_initial_variables(loaded_model, objective_coefficients)
    for i, var in enumerate(variables):
     var.setInitialValue(initial[i])
    print(initial)
    objective_expression = lpSum(coeff * var for coeff, var in zip(objective_coefficients, variables))
    mip_problem += objective_expression  # Add objective function without name

    # Add constraints
    for i in range(num_constraints):
        # constraint_values = [int(val) for val in request.form.getlist(f'constraint_{i+1}')]
        constraint_values = [int(request.form[f'constraint_{i+1}_{j+1}']) for j in range(num_variables)]
        constraint_expression = lpSum(val * var for val, var in zip(constraint_values, variables))
        constraint_limit = int(request.form[f'constraint_{i+1}_limit'])  # Convert to integer
        mip_problem += constraint_expression <= constraint_limit, f"Constraint{i+1}"

    # Print the MIP problem formulation
    print("MIP Problem Formulation:")
    print(mip_problem)

    # Solve the problem
    mip_problem.solve()

    # Get the solution
    solution = {var.name: var.varValue for var in mip_problem.variables()}
    print("Solution:")
    print(solution)

    return render_template('final.html', solution=solution)
    
if __name__ == '__main__':
    app.run(debug=True)
