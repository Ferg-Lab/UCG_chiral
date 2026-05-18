import os
import numpy as np
import pandas as pd
import pickle


def write_potentials(xll, ull, fll, xld, uld, fld, filepath):
  
    fmin  = 0.1
    fmax  = 6.0

    #Write to file
    with open(filepath+'/Pair_L-L.table', 'w') as file:
        file.write('# Table Pair_L-L: id, r, potential, force' + '\n')  # Add a newline character between rows
        file.write('\n')
        file.write('L-L\n')
        file.write('N ' + str(len(fll)) + ' R ' + str(fmin) + ' ' + str(fmax))
        file.write('\n\n')
                
        #print(fll.shape[0])
        for i in range(fll.shape[0]):
            
            #print(xll)
            file.write(str(i+1) + ' ' + str(xll[i]*1) + ' ' + str(ull[i]) + ' ' + str(fll[i]))
            file.write('\n')
                    
    # Open the file in write mode
    with open(filepath+'/Pair_L-D.table', 'w') as file:
        file.write('# Table Pair_L-D: id, r, potential, force' + '\n')  # Add a newline character between rows
        file.write('\n')
        file.write('L-D\n')
        file.write('N ' + str(len(fld)) + ' R ' + str(fmin) + ' ' + str(fmax))
        file.write('\n\n')
                
        for i in range(fld.shape[0]):
            file.write(str(i+1) + ' ' + str(xld[i]*1) + ' ' + str(uld[i]) + ' ' + str(fld[i]))
            file.write('\n')
                    
    # Open the file in write mode
    with open(filepath+'/Pair_D-D.table', 'w') as file:
        file.write('# Table Pair_D-D: id, r, potential, force' + '\n')  # Add a newline character between rows
        file.write('\n')
        file.write('D-D\n')
        file.write('N ' + str(len(fll)) + ' R ' + str(fmin) + ' ' + str(fmax))
        file.write('\n\n')
                
        for i in range(fll.shape[0]):
            file.write(str(i+1) + ' ' + str(xll[i]*1) + ' ' + str(ull[i]) + ' ' + str(fll[i]))
            file.write('\n')        
        
                
def save_models(model_ll, model_ld, filename="models.pkl"):
    """
    Save both spline models into a single pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump({"model_ll": model_ll, "model_ld": model_ld}, f)
    print(f"Models saved to {filename}")


def load_models(filename="models.pkl"):
    """
    Load both spline models from a pickle file.
    Returns: (model_ll, model_ld)
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Models loaded from {filename}")
    return data["model_ll"], data["model_ld"]

def save_coeffs(c_ll, c_ld, sigmoid, filename="coeff.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"c_ll": c_ll, "c_ld": c_ld, "sigmoid": sigmoid}, f)
    print(f"Coeff saved to {filename}")


def load_coeffs(filename="coeff.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["c_ll"], data["c_ld"], data["sigmoid"]

def save_loss(loss, filename="loss.pkl"):
    with open(filename, "wb") as f:
         pickle.dump({"loss": loss}, f)
    print(f'Loss saved')

def load_loss(filename="loss.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data['loss']








