import pandas as pd
def cleandataframe(path):
    dataframe = pd.read_csv(path)
    return(dataframe)