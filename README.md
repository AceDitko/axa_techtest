# axa_techtest

The work in this repo was completed as part of a technical test, using the [kaggle depression dataset](https://www.kaggle.com/datasets/anthonytherrien/depression-dataset/data).
The code within this repo is a solution to the second use case shared in the task brief: Predicting Mental Health. 

This work was completed over the course of the evenings of the 28th and 29th of September (as well as into the early morning of the 30th).   
The initial exploration and analysis was performed in `notebooks/eda.ipynb` and then packaged into the source code files available in the `src/` directory.   
Example tests can be found in the `tests/` directory and can be run with `pytest`.   

### Usage

The code requires that the kaggle depression dataset be saved in a directory named `data/` with the name `depression_data.csv`. This makes the relative path from the top level directory to the file `data/depression_data.csv`. 

All commands to run the source code should be run in the top level directory.   

To train the model, first run `python -m src.train`.   

Then, to obtain a prediction, provide a `.csv` file (ideally in the same `data/` directory as the `depression_data.csv` file) containing the same header rows as `depression_data.csv` and the rows of data you wish to predict.   
Prediction takes the following format: 

`python -m src.predict <input_file.csv> --output_path <output_file.csv> --threshold <threshold>`    

where `--output_path` and `--threshold` are optional arguments. 

##### Arguments: 

 * `input_file.csv`: Path to the input csv file containing the data to be predicted
 * `--output_path`: Optional output path to write the predictions to as a csv. If blank, predictions will be printed to the terminal. 
 * `--threshold`: Optional argument to adjust the threshold at which a prediction is counted as a positive e.g. a probability prediction of 0.4 would be counted as a positive with a threshold of 0.3 but would be counted as a negative with a threshold of 0.5. If blank, the default value of 0.3 will be used. 