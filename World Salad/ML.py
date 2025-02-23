import pandas as pd
import numpy as np


def train():
    df = pd.read_csv("Animal Dataset.csv")

    df["Animal"].value_counts()

    df["Animal_code"] = df["Animal"].astype("category").cat.codes

    df_n = pd.DataFrame({'Rango': ['75-80', '60-70', '90-100', '50-55', 'Up to 120', 'Less than 30']})

    df['Height (cm)'] = df['Height (cm)'].apply(lambda x: int(x.split('-')[0]) if '-' in x and x.split('-')[0].isdigit() else None)

    df['Weight (kg)'] = df['Weight (kg)'].apply(lambda x: int(x.split('-')[0]) if '-' in x and x.split('-')[0].isdigit() else None)

    df['Lifespan (years)'] = df['Lifespan (years)'].apply(lambda x: int(x.split('-')[0]) if '-' in x and x.split('-')[0].isdigit() else None)

    df["Social Solitary"] = (df["Social Structure"]  == 'Group-based').astype(int)

    df['Top Speed (km/h)'] = df['Top Speed (km/h)'].apply(lambda x: x[:2] if isinstance(x, str) else None)

    df['Top Speed (km/h)'] = df['Top Speed (km/h)'].apply(lambda x: str(x).split('.')[0][:2] if isinstance(x, str) else None)

    df['Top Speed (km/h)'] = df['Top Speed (km/h)'].replace(["No","4-","Va","8-"], 0)

    df['Top Speed (km/h)'].astype(int)

    df["Endangered"] = (df["Conservation Status"] == "Endangered").astype(int)

    df["diet_code"] = df["Diet"].astype("category").cat.codes

    X = df[[ "Height (cm)", "Weight (kg)", "diet_code", "Social Solitary"]]

    y = df["Endangered"] 


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12, test_size=0.3) 

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(); 

    rf.fit(X_train, y_train) 

    y_pred = rf.predict(X_test)



    from sklearn.metrics import classification_report 

    print(classification_report(y_test, y_pred)) 

    diet_map = {diet: code for code, diet in enumerate(df['Diet'].astype('category').cat.categories)}

    test = pd.DataFrame( [[200, 200, 5, 0]] , columns = [ "Height (cm)", "Weight (kg)", "diet_code", "Social Solitary"]) 

    rf.predict(test) 



def filter1(animal, df_full) :
    return (df_full[df_full["Animal"] == animal])

def prediction(h, w, d, s) :
    df = pd.read_csv("Animal Dataset.csv")
    diet_map = {diet: code for code, diet in enumerate(df['Diet'].astype('category').cat.categories)}
    h 
    w
    d_n = diet_map[d] 
    s
    df = pd.read_csv("Animal Dataset.csv")

    df["Animal"].value_counts()

    df["Animal_code"] = df["Animal"].astype("category").cat.codes

    df_n = pd.DataFrame({'Rango': ['75-80', '60-70', '90-100', '50-55', 'Up to 120', 'Less than 30']})

    df['Height (cm)'] = df['Height (cm)'].apply(lambda x: int(x.split('-')[0]) if '-' in x and x.split('-')[0].isdigit() else None)

    df['Weight (kg)'] = df['Weight (kg)'].apply(lambda x: int(x.split('-')[0]) if '-' in x and x.split('-')[0].isdigit() else None)

    df['Lifespan (years)'] = df['Lifespan (years)'].apply(lambda x: int(x.split('-')[0]) if '-' in x and x.split('-')[0].isdigit() else None)
    df["Social Solitary"] = (df["Social Structure"]  == 'Group-based').astype(int)
    df['Top Speed (km/h)'] = df['Top Speed (km/h)'].apply(lambda x: x[:2] if isinstance(x, str) else None)
    df['Top Speed (km/h)'] = df['Top Speed (km/h)'].apply(lambda x: str(x).split('.')[0][:2] if isinstance(x, str) else None)
    df['Top Speed (km/h)'] = df['Top Speed (km/h)'].replace(["No","4-","Va","8-"], 0)
    df['Top Speed (km/h)'].astype(int)
    df["Endangered"] = (df["Conservation Status"] == "Endangered").astype(int)
    df["diet_code"] = df["Diet"].astype("category").cat.codes

    X = df[[ "Height (cm)", "Weight (kg)", "diet_code", "Social Solitary"]]

    y = df["Endangered"] 


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12, test_size=0.3) 

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(); 

    rf.fit(X_train, y_train) 

    y_pred = rf.predict(X_test)



    from sklearn.metrics import classification_report 

    print(classification_report(y_test, y_pred)) 

    diet_map = {diet: code for code, diet in enumerate(df['Diet'].astype('category').cat.categories)}

    test = pd.DataFrame( [[200, 200, 5, 0]] , columns = [ "Height (cm)", "Weight (kg)", "diet_code", "Social Solitary"]) 

    rf.predict(test) 



    
    test = pd.DataFrame([[h, w, d_n, s ]], columns = [ "Height (cm)", "Weight (kg)", "diet_code", "Social Solitary"]) 
    pred = rf.predict(test)
    return pred[0] 