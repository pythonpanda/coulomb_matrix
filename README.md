# Sorted Coulomb matrix generator
[Coulomb matrix](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301) has been developed as a descriptor for molecules, inorder to learn and predict their properties using Machine Learning. 

A Python based code to construct a Sorted Coulomb matrix  from SMILES string  of molecules. By default the Sorted Coulomb matrix is saved to a CSV output file containing LabeledPoint vectors optimized to be read by Apache Spark. Apache Spark is particularly optimal for handling big data and comes with built in powerful Machine learning library. 

An optional scikit-learn is invoked at the end of the script to classify molecules using SVM.




