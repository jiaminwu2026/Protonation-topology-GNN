import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.feature_selection import VarianceThreshold
import random


ID = random.randint(0,10000)
HP_TUNING = 'on'

df = pd.read_csv(f'../data/smiles.csv')

smiles = df['base']
mols = []
fps = []
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=4096)

for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    mols.append(mol)
    fps.append(np.array(mfpgen.GetFingerprint(mol)))

fps = VarianceThreshold().fit_transform(fps)
descriptors = np.column_stack((df.loc[:,['peptides%','time']],fps))

for output in ['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']:
    df[output] = np.log(df[output])

for target in ['min','mean','max']:
    if target == 'min':
        y = np.min(df.loc[:,['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']],axis=1)
    if target == 'mean':
        y = np.mean(df.loc[:,['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']],axis=1)
    if target == 'max':
        y = np.max(df.loc[:,['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(descriptors, y, train_size=0.8, random_state=25)

    if HP_TUNING == 'on':

        model = RandomForestRegressor(random_state=25)

        space  = [Integer(1, 200, name='max_depth'),
                Integer(1, 200, name='n_estimators'),
                Integer(2, 50, name='min_samples_split'),
                Integer(1, 50, name='min_samples_leaf')]

        @use_named_args(space)
        def objective(**params):
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1,
                                            scoring="neg_mean_absolute_error"))

        res_gp = gp_minimize(objective, space, n_calls=250, random_state=25, verbose=True)

        print(f'Best: max_depth = {res_gp.x[0]}, n_estimators = {res_gp.x[1]}, min_samples_split = {res_gp.x[2]}, min_samples_leaf{res_gp.x[3]}')

        tuned_model = RandomForestRegressor(max_depth=res_gp.x[0], n_estimators=res_gp.x[1],min_samples_split=res_gp.x[2], min_samples_leaf=res_gp.x[3], random_state=25).fit(X_train,y_train)

    else:
        tuned_model = RandomForestRegressor().fit(X_train,y_train)

    train_pred = tuned_model.predict(X_train)
    test_pred = tuned_model.predict(X_test)

    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(y_train, y_train, color='grey', linewidth=1)
    plt.scatter(y_train,train_pred, 
                label=f'Train R$^2$ {r2_score(y_train,train_pred):.2f} MSE {mean_squared_error(y_train,train_pred):.2f}', 
                color='midnightblue', 
                edgecolors='white', linewidths=0.5, s=50)
    plt.scatter(y_test,test_pred, 
                label=f'Test R$^2$ {r2_score(y_test,test_pred):.2f} MSE {mean_squared_error(y_test,test_pred):.2f}', 
                color='firebrick', 
                edgecolors='white', linewidths=0.5, s=50)
    plt.legend(frameon=False)
    plt.xlabel('Measured Ln(Max Radiance)')
    plt.ylabel('Predicted Ln(Max Radiance)')
    plt.title(f'{target} Ln(Max Radiance)')
    plt.show()

