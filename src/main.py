from model import model
from read_data import read_data
import pandas as pd
import numpy as np

def preprocess(data):
    data = pd.DataFrame(data)
    data = data.replace("", np.nan) # convert empty strings to nan for fillna()
    data = data.fillna({"job": "unknown"})
    data = data.fillna({"education": "tertiary"})
    data = data.fillna({"contact": "telephone"})
    data = data.fillna({"poutcome": "other"})
    data = pd.get_dummies(data, columns=["job", "marital", "education", "contact", "poutcome"], dtype=int)
    print(f"{data}")
    return data


def main():
    model()
    data = read_data("dataset/07_bank_marketing_train.csv")
    data = preprocess(data)
    print("Hello, world\n")

if __name__=="__main__":
    main()
