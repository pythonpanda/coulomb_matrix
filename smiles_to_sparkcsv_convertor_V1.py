#!/data/ganesh/Software/anaconda/bin/python -tt    

# import modules used here 
from __future__ import print_function
import sys
import subprocess
import pybel
import csv
from time import time
from numpy import *
from numba import jit
from sklearn import svm


"""
kernprof -l script.py
python -m line_profiler script.py.lprof
Uncomment @profile after identifying the bottleneck!
"""

'@profile' 


""""

Prepares Input for a  Spark MLlib based Molecule Classifier

Save the training set and test set in CSV format
The first item of each line in CSV should correspond to the category number E.g. '1','C','CC',...., where 1 is the classification number followed by the SMILESTRING.
Run the script as : $./script.py 'trainingset.csv' 'testset.csv'

""""

def extract_csv(file):
    """   
     A function to process the CSV files and import them as python list . 
     Each row of the list inturn forms a sub-list for each class of molecules.
    """
    opened_file = open(file)
    read_csv = csv.reader(opened_file)
    csv_to_list = list(read_csv)
    #csv_to_list
    opened_file.close()
    return csv_to_list


def largest_molecule_size(training_input) :
    '''
    All the rows of Coulomb matrix shoud be of same dimension. Hence we need number of atoms in the largest molecule . 
    This function uses Pybel to compute just that!
    '''    
    mols = [pybel.readstring("smi", molecule)  for rows in training_input for index, molecule in enumerate(rows) if index!=0 ]
    [mol.OBMol.AddHydrogens() for mol in mols]
    return int(max([ len(mol.atoms)  for mol in mols]))

def process_smile(row,par):    
    """
    A function to convert SMILESTRING to 3D coordinates using  openbabel
    """
    dict_list= []
    atomnum_row_array = range(len(row)-1)
    for ind,item in enumerate(row):
        
        if ind !=0:
                        
            cmd =' obabel -:'+str(row[ind])+'    -oxyz -O  '+par+'/'+str(ind)+'_'+str(row[0])+'_'+par+'.xyz   --gen3d'
            output = subprocess.check_output(cmd,stderr=subprocess.STDOUT, shell=True)
            
            dict = {row[0] :par+'/'+str(ind)+'_'+str(row[0])+'_'+par+'.xyz'}
    print("A total of : %d molecules of class : %d converted by OpenBabel"%(int(ind),int(row[0]) ))    

def periodicfunc(element):
    """
    A function to output atomic number for each element in the periodic table
    """
    f = open("pt.txt")
    atomicnum = [line.split()[1] for line in f if line.split()[0] == element]
    f.close()
    return int(atomicnum[0])

def coulombmat(file,dim):
    """
    This function takes in an xyz input file for a molecule, number of atoms in the biggest molecule  to computes the corresponding coulomb Matrix 
    """
    xyzfile=open(file)
    xyzheader = int(xyzfile.readline())
    xyzfile.close()
    i=0 ; j=0    
    cij=zeros((dim,dim))
    chargearray = zeros((xyzheader,1))
    xyzmatrix = loadtxt(file,skiprows=2,usecols=[1,2,3])
    atominfoarray = loadtxt(file,skiprows=2,dtype=str,usecols=[0])
    chargearray = [periodicfunc(symbol)  for symbol in atominfoarray]
    
    for i in range(xyzheader):
        for j in range(xyzheader):
            if i == j:
                cij[i,j]=0.5*chargearray[i]**2.4
            else:
                dist= linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])              
                cij[i,j]=chargearray[i]*chargearray[j]/dist   
    return  cij
 
def matsort(xyzfile,dim):
    """
    Takes in a Coloumb matrix of (mxmn) dimension and peforms a rowwise sorting such that |C(j,:)| > |C(j+1,:)|, J= 1,.......,(m-1)
    Finally returns a vecotrized (m*n,1) column matrix .
    """   
    unsorted_mat = coulombmat(xyzfile,dim)
    summation = array([sum(x**2) for x in unsorted_mat])
    sorted_mat = unsorted_mat[argsort(summation)[::-1,],:]    
    return sorted_mat.ravel()

# Gather our code in a main() function
def main():
    ########################### Reading Inputs & Preparing Folders  ##############################################################
    start = time()
    print('\nThe training data is read from :  %s \n  The test data is read from:%s '  %(sys.argv[1],sys.argv[2]))
    print('\nRemoving Exisiting Training and Test xyz directories')
    output = subprocess.call('rm -fr train test ', shell=True)    
    
    print('\nCreating new training and test directories')
    output = subprocess.check_output('mkdir train test ',stderr=subprocess.STDOUT, shell=True)            
    print(output)        
    
   ########################### Pre-processing CSV inputs  ##############################################################
    print('\nExtracting and analyzing CSV Data \n')
    training_input = extract_csv(sys.argv[1])                   # sys.argv[0] is the script name itself and can be ignored
    test_input = extract_csv(sys.argv[2])
    
    # Process Training set data and get the dimension for Coulomb matrix
    no_of_class = len(training_input)
    print('\nThe training set has  %d classes of molecules to train' %(no_of_class ) )
    
    max_atom_index = largest_molecule_size(training_input)      # Dimension of rows of the Coulomb matrix
    print('\nThe largest molecule has: %d atoms' %(max_atom_index ) )     
    
   ########################### Post-processing CSV training inputs  ##############################################################
    print('\nPost-processing CSV training set data to generate matrices for training set\n')
    par='train'    
    train_iter_array = range(no_of_class)
    for iter, row in enumerate(training_input):
        par='train'
        train_iter_array[iter] = len(row)-1
        process_smile(row,par)
        print('\n ')        
    
    q=0
    scikit_train_Xarray = empty((sum(train_iter_array),max_atom_index*max_atom_index))
    scikit_train_Yarray = empty(sum(train_iter_array))
    for classes in range(no_of_class):
        for subclass in range(train_iter_array[classes]):            
            label=array([float(classes)])            
            filetrain=open('train_array.csv','a')
            training_sarray = matsort(par+'/'+str(subclass+1)+'_'+str(classes+1)+'_'+par+'.xyz',max_atom_index)   
            scikit_train_Xarray[q] = training_sarray
            scikit_train_Yarray[q] = label
            save_train_array = concatenate((label, training_sarray), axis=0)            
            savetxt(filetrain,save_train_array[None],fmt='%.6f',delimiter=',',newline='\n')                #numpyarray[None] should be used to avoid error write all elements are columns in output file!
            filetrain.close()
            q += 1
    print("The sorted Coloumb Matrix (vectorized) for the training  set has been written to : 'train_array.csv' \n")        

    ########################### SVC-SCIKIT_LEARN  ##############################################################
    print('\n Learning from the Training set data \n')
    clf = svm.SVC()
    clf.fit(scikit_train_Xarray, scikit_train_Yarray)      
   
   ########################### Post-processing CSV test inputs  ##############################################################
    print('\nPost-processing CSV  data to generate matrices for test set\n')
    par='test'
    test_iter_array = range(no_of_class)                                  # An array to store the number of test sets in each classifying  groups E.g. class 1 has 15 molecules so test_iter_array[0] = 15
    print('\nPost-processing CSV test set data to generate matrices \n')
    for iter, row in enumerate(test_input):
        par='test'  
        test_iter_array[iter] = len(row)-1
        process_smile(row,par)        
        print('\n')

    r=0
    scikit_test_Xarray = empty((sum(test_iter_array),max_atom_index*max_atom_index))
    scikit_test_Yarray = empty(sum(test_iter_array))    
 
    for classes in range(no_of_class):
        for subclass in range(test_iter_array[classes]):
            label=array([float(classes)])
            filetest=open('test_array.csv','a')
            test_sarray = matsort(par+'/'+str(subclass+1)+'_'+str(classes+1)+'_'+par+'.xyz',max_atom_index)            
            scikit_test_Xarray[r] = test_sarray
            scikit_test_Yarray[r] = label
            save_test_array = concatenate((label, test_sarray), axis=0)            
            savetxt(filetest,save_test_array[None],fmt='%.6f',delimiter=',',newline='\n')
            filetest.close()
            r += 1 
                        
    print("The sorted Coloumb Matrix (vectorized) for the test set has been written to : 'test_array.csv' \n")
    print('\nNote : First element of matrix for each molecule correponds to the label point for supervised learning')        
    
   ########################### SVC-SCIKIT_CV or TEST ##############################################################
    print('\n Validating the  Test set data \n')
    prediction = clf.predict(scikit_test_Xarray)
    print(prediction == scikit_test_Yarray)
    success = 100.*sum(prediction == scikit_test_Yarray)/float(len(scikit_test_Yarray))
    print("\n The SVM predictions are  %.4f %% accurate" %(success) )

    end = time()
    print("\nTotal execution time was %.4f seconds" %(end-start) )

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
