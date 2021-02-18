# Code

Use this folder for the code related to your project.

caption_beam_new.ipynb
- For generating and showing image captions using beam search

checkpoints.py
- All checkpoints names and vocabs stored in this file

eval-run.py
- Running the evaluation process (Device, model name and vocab should be specified before running) 

utils.py
- Public codes and functions for the whole project (Device should be the same as other files)

eval.py
- Functions used for evaluation process

train.py
- Running the training process (Device, hyper-parameters, model name and vocab should be specified before running)

datasets.py
- Codes for retrieving data from the datasets

caption_greedy_new.ipynb
- For generating and showing image captions using greedy search

models.py
- The model used for this project (Device, dimensions should be specified before training. When training using pre-trained VGG10, encoder dimension should be modified)

create_input_files.py
- To retrive data from the database and create input files

caption.py
- To retrieve captions from the input files.