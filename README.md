# Decision-Tree-Classifier

This project is an implementation of a CART based Decision Tree Classifier. 5 different datasets from [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets.html) 

### Files:
- dtClassifier.py - definitions for the partition, find best-split and algorithm evaluation with datasets
- leaf.py - definition node for a leaf as a predicted class
- decision.py - definition node for a query on the parse tree

### dtClassifier.py command line options:
        
    -h, --help            show this help message and exit
    
    --dataset SET_TYPE    Type 'seismic' for seismic-bumps dataset OR 
                          'banknote' for banknote-authentication dataset OR 
                          'bankruptsy' for bankrupty qualitative parameters OR 
                          'balance' for balance-scale dataset OR 
                          'weather' for the weather dataset
    
    --dt                  Print the decision tree | default=false
    
    --kf N_FOLDS          No. of K folds for cross validation | default=3

    example: python dtClassifier.py --dataset seismic --kf 5 --dt

