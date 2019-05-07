# Deep Onion Fingerprinting

# Models
1) RNN based model
2) CNN based model
3) All CNN based model

# Prerequisites
1) Anaconda
2) Python 3.x

# How to run

Create environment
```bash
> conda env create -f environment.yml
```
Activate environment
```bash
> conda activate deep-cf
```
Run the program
```bash
> python3 src/circuit_classifier.py
``` 

# Results

## Accuracies
|            | All-CNN | CNN | RNN |
|------------|---------|-----|-----|
|**Undefended**  | 77.4    |81.6 | 2.0 |
|**Tamarow**     | 6.6     | 7.6 | 2.0 |
|**WTF-PAD**     | 29.6    |44.4 | 2.0 |

