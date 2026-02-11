# KT_Benchmark Codes

This is an additional file implemented after existing [benchmark](https://github.com/theophilegervet/learner-performance-prediction)

## How to use

These files are provided with submodules removed for anonymous review. After cloning the original repository, place these files into their corresponding paths and the code will run correctly. This folder follows the same directory structure as the original repository.

### train_priorKT.py
Prior Knowledge Tracing is implemented here.  
To train PriorKT:
```bash
python train_priorKT.py --dataset <dataset codename>
```
### train_saint.py

We implemented SAINT to analyze predictions from larger deep learning models. The implementation follows the original paper. Since some datasets are missing timestamps and require time labels for SAINT+, we implemented the base version rather than SAINT+.
To train SAINT:

```bash
python train_saint.py --dataset <dataset codename>
```
### analyze_kt_dependency.py

Code use for regression analsis. to run this, you need to run PFA first following the [original instruction](https://github.com/theophilegervet/learner-performance-prediction)

### Eedi dataset

This was added to enable prediction experiments on a larger dataset. We used the datasets from the NeurIPS 2020 Eedi Challenge (1/2). Due to the dataset size, an external storage source is normally required; however, for anonymization purposes, sample data is provided instead.