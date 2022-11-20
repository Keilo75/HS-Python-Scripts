## `clean.py`

Walks over a given folder recursively and transforms each image:

- Removes transparency
- Resizes to set size
- Removes the background
- Changes the file format

## `train.py`

Trains a model given a configuration:

- Loads and augment data
- Creates model and compute hash
- Checks if hash has been trained before
- Trains model x times
- Saves best result to `/models` folder and creates plots
- Adds result to csv

# Related

[ImgCollect](https://github.com/Keilo75/HS-ImgCollect)
