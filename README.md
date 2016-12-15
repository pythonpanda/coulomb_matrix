# Sorted Coulomb matrix generator
[Coulomb matrix](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301) has been developed as a descriptor for molecules, inorder to learn and predict their properties using Machine Learning. 

[A Python code](https://github.com/pythonpanda/coulomb_matrix/blob/coulomb-matrix-generator/smiles_to_sparkcsv_convertor_V1.py) to construct a Sorted Coulomb matrix  from SMILES string  of molecules. By default the Sorted Coulomb matrix is saved to a CSV output file containing LabeledPoint vectors optimized to be read by [Apache Spark](http://spark.apache.org/). Apache Spark is particularly optimal for handling big data and comes with built in [powerful Machine learning library](http://spark.apache.org/mllib/). 

An optional [scikit-learn](http://scikit-learn.org/stable/) is invoked at the end of the script to classify molecules using SVM.




