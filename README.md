# coulomb_matrix
A Python based code to construct a Sorted Coulomb matrix from SMILES (CSV input) of molecules.
By default the Sorted Coulomb matrix is saved to a CSV output file containing LabeledPoint easily read by Apache Spark.
An optional scikit-learn is invoked at the end of the script to classify molecules using SVM.
