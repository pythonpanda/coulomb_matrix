# [Sorted Coulomb matrix](http://www.seas.harvard.edu/sites/default/files/files/Undergraduate%20Program%20in%20Computer%20Science/Sun.pdf) generator
[Coulomb matrix](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301) has been developed as a descriptor for molecules, inorder to learn and predict their properties using Machine Learning. 

[A Python script](https://github.com/pythonpanda/coulomb_matrix/blob/coulomb-matrix-generator/smiles_to_sparkcsv_convertor_V1.py) to construct a Sorted Coulomb matrix  from [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) string  of molecules. The code internally utilizes [openbabel](http://openbabel.org/wiki/Main_Page) to process the chemical data input in the form of SMILES. By default the Sorted Coulomb matrix is saved to a CSV output file containing LabeledPoint vectors optimized to be read by [Apache Spark](http://spark.apache.org/). Apache Spark is particularly optimal for handling big data and comes with built in [powerful Machine learning library](http://spark.apache.org/mllib/). 

An optional [scikit-learn](http://scikit-learn.org/stable/) is invoked at the end of the script to classify molecules using SVM.




