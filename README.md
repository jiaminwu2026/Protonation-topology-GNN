# ML repository for GNN predictions

Graph neural network (GNN) models for predicting the ln(max radiance) of branched peptides, supporting the manuscript "Protonation topology engineering for intracellular RNA delivery."

## Requirements

- Python 3.12.4
- PyTorch 2.2.2
- PyTorch Geometric 2.5.3 (GNNExplainer via `torch_geometric.explain.Explainer`)
- scikit-learn 1.5.1
- scikit-optimize
- RDKit 2024.3.5
- ChemProp (for the atomwise directed-MPNN benchmark)

Hardware: [TO BE CONFIRMED — CPU vs GPU; update once confirmed].

## Installation

    git clone https://github.com/jiaminwu2026/Protonation-topology-GNN.git
    cd Protonation-topology-GNN
    pip install -r requirements.txt

Typical install time: [TO BE CONFIRMED — e.g. a few minutes on a standard desktop].

## Data

Data used for training and testing. `bps` denotes the points on the `base` peptide chain in which the `branches` are found. Dataset generation requires these bps to be set. This can be time-consuming for larger datasets, so we recommend modifying the code to pickle a dataset object for easy re-use.

## Results

Compiled results from the ML study. `VS_results_XXX` are tabulated predictions for each set of virtually screened peptides. `models` contains pre-trained model weights for reproduction.

## Tests

Scripts for creating and testing models.

- **Training:** run `GCN.py`. This performs an 80:20 train/validation split, trains on the training set, and predicts a held-out validation set.
- **Inference:** load a pre-trained model from `Results/models` and apply it to the virtual-screening (VS) data.

## Demo / expected output

Running inference with the provided pre-trained weights on the included VS data reproduces the `VS_results_XXX` prediction tables. Expected run time: [TO BE CONFIRMED — e.g. training ~X min; inference < 1 min].

## Model summary

Min, mean, and maximum ln(max radiance) are modeled separately. The GNN aggregates a base-chain/branch-point representation and passes it to a final MLP regressor. Generalizability was assessed by leave-one-peptide-out cross-validation, and the approach was benchmarked against Morgan-fingerprint random forests and an atomwise directed-MPNN (ChemProp).

## License

Code is available for peer review and academic use. Commercial use / licensing is subject to the UBC technology transfer office. A patent application covering the method has been filed.
