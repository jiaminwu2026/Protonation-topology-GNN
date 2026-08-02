import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
from chemprop.models.utils import save_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

num_workers = 0
df = pd.read_csv(f'../data/smiles.csv')

MAX_EPOCHS = 7000

smiles_column = 'base' 

smis = df.loc[:, smiles_column].values

for output in ['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']:
    df[output] = np.log(df[output])

for target in ['max']:
    if target == 'min':
        ys = np.min(df.loc[:,['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']],axis=1).values.reshape(-1,1)
    if target == 'mean':
        ys = np.mean(df.loc[:,['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']],axis=1).values.reshape(-1,1)
    if target == 'max':
        ys = np.max(df.loc[:,['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']],axis=1).values.reshape(-1,1)
    

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    for idx, _ in enumerate(all_data):
        _.x_d = df.iloc[idx,[2,3]].values.astype(float)


    mols = [d.mol for d in all_data]  

    train_indices, val_indices = train_test_split(range(len(all_data)), test_size=0.2, random_state=25)

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices=None
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, batch_size=32)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)

    mp = nn.BondMessagePassing()
    agg = nn.SumAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim+2, output_transform=output_transform, hidden_dim=514)
    batch_norm = False

    metric_list = [nn.metrics.MSEMetric(), nn.metrics.MAEMetric()]

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator="auto",
        devices="auto",
        max_epochs=MAX_EPOCHS, 
    )

    trainer.fit(mpnn, train_loader, val_loader)

    with torch.inference_mode():
        train_preds = trainer.predict(mpnn, train_loader)
        val_preds = trainer.predict(mpnn, val_loader)

    train_preds = np.concatenate(train_preds, axis=0)
    test_preds = np.concatenate(val_preds, axis=0)


    train_meas = [ys[i] for i in train_indices]
    test_meas = [ys[i] for i in val_indices]


    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(train_meas, train_meas, color='grey', linewidth=1)
    plt.scatter(train_meas,train_preds, 
                label=f'Train R$^2$ {r2_score(train_meas,train_preds):.2f} MSE {mean_squared_error(train_meas,train_preds):.2f}', 
                color='midnightblue', 
                edgecolors='white', linewidths=0.5, s=50)
    plt.scatter(test_meas,test_preds, 
                label=f'Test R$^2$ {r2_score(test_meas,test_preds):.2f} MSE {mean_squared_error(test_meas,test_preds):.2f}', 
                color='firebrick', 
                edgecolors='white', linewidths=0.5, s=50)
    plt.legend(frameon=False)
    plt.xlabel('Measured Ln(Max Radiance)')
    plt.ylabel('Predicted Ln(Max Radiance)')
    plt.title(f'{target} Ln(Max Radiance)')
    plt.savefig(f'../results/ChemProp_{target}.png')
    plt.clf()

    # pd.DataFrame({'train_meas':train_meas,'train_preds':train_preds,'test_meas':test_meas,'test_preds':test_preds}).to_csv(f'../results/ChemProp_{target}.csv')
