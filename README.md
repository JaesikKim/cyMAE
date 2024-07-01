# Cytometry Masked Autoencoder: An Accurate and Interpretable Automated Immunophenotyper (cyMAE)

This project introduces the Cytometry Masked Autoencoder (cyMAE), an automated solution for immunophenotyping tasks, including cell type annotation, using high-throughput single-cell cytometry data. The cyMAE model is specifically designed to work with the MDIPA panel, and it is crucial to follow the specified input marker order for accurate results. Additionally, it is recommended to use quality-controlled data to ensure the best performance of the model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training from scratch](#Training-from-scratch)
- [Citation](#Citation)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/JaesikKim/cyMAE.git
    cd cyMAE
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda env create -n cymae -f environment.yml
    conda activate cymae
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Model checkpoints
Download the model checkpoints from [Google Drive](https://drive.google.com/drive/folders/18LvIuLsDhExlbMZi0MdrVnNGgsJsFkcP?usp=drive_link)
- `pretrain_mae_30D_6L_0.25R` : pre-trained
- `cymae_30D_6L_pretrained0.25R` : fine-tuned for cell type annotation

#### Load the cyMAE checkpoint
```python
import torch
from timm.models import create_model
import modeling_finetune

device = "cuda:0" # specify your device (cpu or gpu)
checkpoint = torch.load("cymae_30D_6L_pretrained0.25R_fold0_0.0064lr_200epoch_checkpoint-best.pth", map_location=torch.device(device))
args = checkpoint['args']
model = create_model(
    args.model,
    pretrained=False,
    num_classes=args.nb_classes,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    attn_drop_rate=args.attn_drop_rate,
    drop_block_rate=None,
    use_mean_pooling=args.use_mean_pooling,
    init_scale=args.init_scale,
    ).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()
```
    
### Quick Start (cell type annotation)

```python
idx_to_class = {0: 'Plasmablast', 1: 'Th2/activated', 2: 'Treg/activated', 3: 'CD8Naive', 4: 'Treg', 5: 'EarlyNK', 6: 'CD66bnegCD45lo', 7: 'CD4Naive', 8: 'Th2', 9: 'CD8TEM2', 10: 'Th17', 11: 'IgDposMemB', 12: 'CD8Naive/activated', 13: 'CD8TEMRA/activated', 14: 'Eosinophil', 15: 'CD8TEM3/activated', 16: 'DPT', 17: 'MAITNKT', 18: 'gdT', 19: 'CD8TEM2/activated', 20: 'nnCD4CXCR5pos/activated', 21: 'IgDnegMemB', 22: 'CD45hiCD66bpos', 23: 'LateNK', 24: 'Neutrophil', 25: 'DNT', 26: 'Basophil', 27: 'pDC', 28: 'CD8TEM1/activated', 29: 'mDC', 30: 'Th1', 31: 'DNT/activated', 32: 'Th1/activated', 33: 'CD8TEMRA', 34: 'CD8TCM/activated', 35: 'CD8TEM1', 36: 'CD4Naive/activated', 37: 'NaiveB', 38: 'ILC', 39: 'CD8TEM3', 40: 'Th17/activated', 41: 'CD8TCM', 42: 'ClassicalMono', 43: 'DPT/activated', 44: 'nnCD4CXCR5pos', 45: 'TotalMonocyte'}

# input marker order should follow this
# ['89Y_CD45', '141Pr_CD196_CCR6', '143Nd_CD123_IL-3R', '144Nd_CD19', '145Nd_CD4', 
# '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD16', '149Sm_CD45RO', '150Nd_CD45RA', 
# '151Eu_CD161', '152Sm_CD194_CCR4', '153Eu_CD25_IL-2Ra', '154Sm_CD27', '155Gd_CD57', 
# '156Gd_CD183_CXCR3', '158Gd_CD185_CXCR5', '160Gd_CD28', '161Dy_CD38', '163Dy_CD56_NCAM', 
# '164Dy_TCRgd', '166Er_CD294', '167Er_CD197_CCR7', '168Er_CD14', '170Er_CD3', 
# '171Yb_CD20', '172Yb_CD66b', '173Yb_HLA-DR', '174Yb_IgD', '176Yb_CD127_IL-7Ra']

input_data = torch.tensor([your_input_data]).to(device) # torch.tensor (C, 30)
with torch.no_grad():
preds = model(input_data)
preds = torch.max(preds,1)[1]
preds = [idx_to_class[idx.item()] for idx in preds]
print(preds)
```

### Tutorials
TBA

## Training from scratch
If you want to use cyMAE with a different panel, you need to train the model from scratch. We provide the training and evaluation codes. Follow the instructions below:

### Input format for Dataloader
A generic data loader where the samples are arranged as follows:
```bash
root/xxx.fcs
root/xxy.fcs
root/xxz.fcs
root/_FilterValuesToNames.csv
root/meta.csv
```

* FCS file requirements:
    * Modify the feature list in `dataset_folder.py` at line 51 to include your specific markers.
    ```python
    ['marker1', 'marker2', ..., 'markerN' 'OmiqFilter']
    ```
    * For fine-tuning, the last column, 'OmiqFilter', should represent the cell type ID in integer format.
       
* _FilterValuesToNames.csv:

	* This file should map cell type IDs to their respective names.
	* See Example: `file_examples/example_FilterValuesToNames.csv`

* meta.csv:

	* For fine-tuning, this file should include fold information to split the data into training (0), validation (1), and test (2) sets.
	* See Example: `file_examples/example_meta.csv`

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python run_mae_pretraining.py \
   --data_path path_for_training_set \
   --model pretrain_mae_30D_6L \
   --epochs 250 \
   --warmup_epochs 5 \
   --batch_size 4096 \
   --mask_ratio 0.75 \
   --lr 1.5e-5
```
### Finetuning
```bash
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
   --data_path path_for_finetuning_set \
   --fold 0 \
   --lr 1e-4 \
   --min_lr 1e-6 \
   --nb_classes 46 \
   --epochs 200 \
   --tol 200 \
   --warmup_epochs 2 \
   --batch_size 16384 \
   --model cymae_30D_6L \
   --finetune path_for_pretrained_model.pth
```
### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
    --eval \
    --data_path path_for_finetuning_set \
    --external_data_path path_for_evaluation_set \
    --model cymae_30D_6L \
    --finetune path_for_finetuend_model.pth \
    --nb_classes 46 \
    --batch_size 16384
```

### Dependencies
- python 3 with numpy >= 1.23.5
- pytorch >= 1.13.1
- scikit-Learn >= 1.2.0
- pandas >= 1.5.3
- FlowCytometryTools == 0.5.1
- shap >= 0.45.1
- statsmodels >= 0.14.2
- timm >= 0.6.12

### Citation
```
@article{Kim2024,
    author  = {Kim, Jaesik and Ionita, Matei and Lee, Matthew and McKeague, Michelle L. and Pattekar, Ajinkya and Painter, Mark M. and Wagenaar, Joost and Truong, Van and Norton, Dylan T. and Mathew, Divij and Nam, Yonghyun and Apostolidis, Sokratis A. and Clendenin, Cynthia and Orzechowski, Patryk and Jung, Sang-Hyuk and Woerner, Jakob and Ittner, Caroline A.G. and Turner, Alexandra P. and Esperanza, Mika and Dunn, Thomas G. and Mangalmurti, Nilam S. and Reilly, John P. and Meyer, Nuala J. and Calfee, Carolyn S. and Liu, Kathleen D. and Matthy, Michael A. and Swigart, Lamorna Brown and Burnham, Ellen L. and McKeehan, Jeffrey and Gandotra, Sheetal and Russel, Derek W. and Gibbs, Kevin W. and Thomas, Karl W. and Barot, Harsh and Greenplate, Allison R. and Wherry, E. John and Kim, Dokyoon},
    title   = {Cytometry Masked Autoencoder: An Accurate and Interpretable Automated Immunophenotyper},
    year    = {2024},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/10.1101/2024.02.13.580114v2},
    journal = {bioRxiv}
}
```

### Acknowledgement
This project was adapted from the MAE code found at the following repository. We extend our gratitude for their work.
https://github.com/pengzhiliang/MAE-pytorch

### License
Apache-2.0 license

### Contact
For questions or comments regarding the code or model, please contact:
jaesik.kim@pennmedicine.upenn.edu