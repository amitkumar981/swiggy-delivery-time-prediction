import numpy as np
import pandas as pd
import requests
import os


def get_root_directory() -> str:
    """Get the root directory (one level up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))  
    return root_dir

root_path=get_root_directory()

data_path = os.path.join(root_path,'data','raw','swiggy.csv')

predict_url="http://127.0.0.1:8000//predict"

sample_row=pd.read_csv(data_path).dropna().sample(1)

print('the target value is',sample_row.iloc[:,-1].values.item().replace("(min)",""))

#drop target column
data=sample_row.drop(columns=[sample_row.columns.to_list()[-1]]).squeeze().to_dict()


# get the response from API
response = requests.post(url=predict_url,json=data)

print("The status code for response is", response.status_code)

if response.status_code == 200:
        print(f"The prediction value by the API is {float(response.text):.2f} min")
else:
    print("Error:", response.status_code)






 