# Translation Models on Expert Finding Machine Learning

This repository contains source codes developed for Machine Learning(Word embedding approach) in this paper:

	Arash Dargahi Nobari, Sajad Sotudeh Gharebagh and Mahmood Neshati. “Skill Translation Models in Expert Finding”,
	In proceedings of The 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’17), Aug 2016.

You may check the [paper](http://dl.acm.org/citation.cfm?id=3080719) ([PDF](http://facultymembers.sbu.ac.ir/neshati/wp-content/uploads/2015/03/Skill-Translation-Models-in-Expert-Finding.pdf)) for more information.

Main repository for other codes can be accessed [here](https://github.com/arashdn/sof-expert-finding)


## Requirements

Python3.5 and Tensorflow 0.11 is required for running the code.


## Usage
Many files on this project are just test and backup file. Only the following files are important:
- `deep_transalteion_new.py`: To run algorithm if you don't include bias parameter
- `result.py`: To get final results if you don't include bias parameter
- `deep_transalteion_new_bias.py`: To run algorithm if you include bias parameter
- `result_with_bias.py`: To get final results if you include bias parameter

Before running code you need to generate 1gram.csv file using java app in [this](https://github.com/arashdn/sof-expert-finding) repository and put it into data/grams folder

## Data

All of data is ignored in git repository.

These files can be downloaded from [dropbox](https://www.dropbox.com/s/h3wq1ppdfgq3obu/data_python.zip) This file includes `data` folder

The `data` folder includes the following files and folders:

- `grams`: TF and TF/IDF weight for words.
- `java_a_tag.txt`: Tags for each answer (Answers don't have tag by their self, taged are extracted from related questions)
- `doc_len.txt`: Normalized lenght for each answer.
- `topWords.txt`: top frequent words in answers.


## Citation

Please cite the paper, If you used the codes in this repository.

```
@inproceedings{DargahiNobari:2017:STM,
 author = {Dargahi Nobari, Arash and Sotudeh Gharebagh, Sajad and Neshati, Mahmood},
 title = {Skill Translation Models in Expert Finding},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR '17},
 year = {2017},
 isbn = {978-1-4503-5022-8},
 location = {Shinjuku, Tokyo, Japan},
 pages = {1057--1060},
 numpages = {4},
 url = {http://doi.acm.org.ezp3.semantak.com/10.1145/3077136.3080719},
 doi = {10.1145/3077136.3080719},
 acmid = {3080719},
 publisher = {ACM},
 keywords = {expertise retrieval, semantic matching, stackoverflow, statistical machine translation, talent acquisition},
} 
```
