# from sklearn.datasets import load_breast_cancer


# X,y = load_breast_cancer(as_frame = True, 
#     return_X_y = True)


from sklearn.datasets import make_classification
import pandas as pd

X,y = make_classification(n_samples = 1500, n_features = 45, n_informative = 7, class_sep = 0.3, random_state = 42)
X = pd.DataFrame(X)