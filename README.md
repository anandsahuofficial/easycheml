# EASYCHEML : A simple tool for using Artificial Intelligence in Chemistry
This article belongs to Author: Anand Sahu <anandsahuofficial@gmail.com>

## Installation

It is recommended to create a virtual environment and do everything in it
for the purpose of this tutorial,
so that you won't mess up your python installation.

For Python 3.6+, you may use the `venv` module in the standard library.
[HOWTO](https://docs.python.org/3/library/venv.html#creating-virtual-environments)

For previous versions of Python, you may use [`virtualenv`](https://virtualenv.pypa.io/en/latest/).

After creating the virtual environment,
it might be a good idea to update the base packages we are going to use:
```bash
$ pip install -U easycheml
```
## Usage
For a general machine learning project, one usually starts with Data Preprocessing, then Modelling, thereafter Postprocessing and Visualization of results produced through ML models.

### Step 1: Preprocessing of Data 

```python
from easycheml.preprocessing import PreProcessing as p
import pandas as pd


df=p.preprocess_data(path_data,'LUMO_calculated',['Smiles'])
print(df)
```


### Step 2: Modelling 
### Ste3 3: Postprocessing and Visualization



