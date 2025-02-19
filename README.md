## Dataset

The dataset used in this challenge is already in the repository at [processed_data.csv](/processed_data.csv). 
It is a trimmed version of the dataset from [IMDb Movie Dataset: All Movies by Genre](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre/data). 
To recreate it, just run `python data.py` from the root directory of this repository. 
For clarity, the `nltk` module's data is downloaded at `/nltk_data` for word tokenizing and lemmatization. 

## Setup

Python Version: `3.12.8`
To create a virtual environment and install requirements, 
```
python -m venv env
source env/Scripts/activate
pip install -r requirements.txt
```
in a UNIX environment or Git Bash. 

## Running

To query "A comedic movie about vampires", 
```
python main.py -d "A comedic movie about vampires"
```

You can also specify the number of top recommended movies with the `-c` flag. 
```
python main.py -d "A comedic movie about vampires" -c 3
```

## Results

![out](/sample_output.png)

---

### Salary

At $20 per hour with 30 hours per week, I expect a month's salary to be $2400. 