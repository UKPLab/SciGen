# SciGen data splits

SciGen consists of three settings based on the training data size: (1) few-shot, (2) medium, and (3) large.

The test data is the same in all the settings and it consists of expert annotations.
We split the test set into two `Computation and Language` (CL) and `Other` domains, in which the `Other` domain mainly contains table-description pairs from the `Machine Learning` (ML) articles.


The data splits in `few-shot` only contain table-description pairs from expert annotations. 
The training and development sets in this setting only contain pairs from the `CL` articles.

The training and development sets in the `medium` setting contain those in `few-shot` plus automatically extracted pairs from additional `CL` articles. 

The training and development sets in the `large` setting contain those in `medium` in addition to automatically extracted pairs from additional `ML` articles.    

## Dataset structure

* `dataset/train` -- this folder contains the training data for the few-shot, medium, and large settings
* `dataset/development` -- this folder contains the developments sets for each of the settings
* `dataset/test` -- contains test-Cl.json and test-Other.json test files

## Data format
Each entry in each of the json files represents a table-description and consist of the following fields:

* `paper` -- the title of the corresponding article
* `paper_id` -- the arXiv identifier of the corresponding article
* `table_caption` -- the caption of the table
* `table_column_names` -- the first row of the table, i.e., the header row
* `table_content_values` -- the list of rows, excluding the header row, in the table
* `text` -- the description of the table


## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This dataset is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


