Release 2.1
25th March 2019

Introduction
----------------------

This directory contains the Cambridge English Write & Improve (W&I) corpus and an annotated subset of the LOCNESS corpus which were both released as part of the BEA2019 shared task on Grammatical Error Correction. 

Although these corpora were compiled/annotated especially for the shared task, please see and cite the following paper for more information on the W&I corpus:

Helen Yannakoudakis, Øistein E. Andersen, Ardeshir Geranpayeh, Ted Briscoe and Diane Nicholls. 2018. Developing an automated writing placement system for ESL learners. Applied Measurement in Education, 31:3, pages 251-267.

Also see the following link for more information on the LOCNESS corpus: https://uclouvain.be/en/research-institutes/ilc/cecl/locness.html

These corpora are released for non-commercial research and educational purposes only; please refer to the licences for terms of use. By downloading and using these corpora you agree to these licences.

Corpora
----------------------

-- Write & Improve --

Write & Improve (https://writeandimprove.com/) is an online web platform that assists non-native English students with their writing. Specifically, students from around the world submit letters, stories, articles and essays in response to various prompts, and the W&I system provides instant feedback. Since W&I went live in 2014, W&I annotators have manually annotated some of these submissions and assigned them a CEFR level.

-- LOCNESS -- 

The LOCNESS corpus consists of essays written by native English students. It was originally compiled by researchers at the Centre for English Corpus Linguistics at the University of Louvain. Since native English students also sometimes make mistakes, we asked the W&I annotators to annotate a subsection of LOCNESS so researchers can test the effectiveness of their systems on the full range of English levels and abilities.

-- Statistics --

We release 3,600 annotated submissions to W&I across 3 different CEFR levels: A (beginner), B (intermediate), C (advanced). We also release 100 annotated native (N) essays from LOCNESS.

We attempted to balance the corpora such that there is a roughly even distribution of sentences at different levels across each of the training, development and test sets. Due to time constraints, we are unable to release a native training set from LOCNESS. An overview of the data is shown in the following table:

------|-----------|---------|---------|---------|---------|---------|
Split | Stats     |    A    |    B    |    C    |    N    |  Total  |
------|-----------|---------|---------|---------|---------|---------|
Train | Texts     |   1,300 |   1,000 |     700 |       0 |   3,000 |
      | Sentences |  10,493 |  13,032 |  10,783 |       0 |  34,308 |
      | Tokens    | 183,684 | 238,112 | 206,924 |       0 | 628,720 |
------|-----------|---------|---------|---------|---------|---------|
Dev   | Texts     |     130 |     100 |      70 |      50 |     350 |
      | Sentences |   1,037 |   1,290 |   1,069 |     988 |   4,384 |
      | Tokens    |  18,691 |  23,725 |  21,440 |  23,117 |  86,973 |
------|-----------|---------|---------|---------|---------|---------|
Test  | Texts     |     130 |     100 |      70 |      50 |     350 |
      | Sentences |   1,107 |   1,330 |   1,010 |   1,030 |   4,477 |
      | Tokens    |  18,905 |  23,667 |  19,953 |  23,143 |  85,668 |
------|-----------|---------|---------|---------|---------|---------|
Total | Texts     |   1,560 |   1,200 |     840 |     100 |   3,700 |
      | Sentences |  12,637 |  15,652 |  12,862 |   2,018 |  43,169 |
      | Tokens    | 221,280 | 285,504 | 248,317 |  46,260 | 801,361 |
------|-----------|---------|---------|---------|---------|---------|

Anonymisation
----------------------

Since people sometimes submit personal or identifying information to Write & Improve, especially when writing letters or introducing themselves, we manually anonymised parts of the corpus. Specifically, we changed the following:
    1. Names
    2. Dates of birth
    3. Postal Addresses
    4. Email Addresses / Usernames
    5. Phone Numbers

Whenever we changed names, we tried to substitute the original with names from the same nationality; e.g. Spanish names were replaced with other Spanish names. The names of famous people and public figures were not changed. 
Dates of birth were randomised, and the top lines of addresses were changed to fictional streets and house numbers.
Postcodes were also randomised, although the order of letters and numbers was preserved; e.g. AZ1 2GG -> QQ3 7AY. 
The local-part of email addresses was also changed to a series of random letters and numbers, but the domain was preserved unless it was private.
The area/country/etc. code of phone numbers was preserved and all remaining digits changed to 12345678.

In all cases, we preserved the formatting of the original string. For example lower case names were replaced with other lower case names, and whitespace separated phone numbers were separated with the same number of whitespace characters. Ultimately, we tried to keep the text as close to the original as possible whilst removing information that might allow users or other named persons to be identified.

Data Formats
----------------------

The W&I corpus is available in two different formats: JSON and M2.

-- JSON --
The JSON format is the raw unprocessed version of the corpus. Each line in a JSON file contains the following fields:
    id     : A unique id for the essay.
    userid : A unique id for the user who submitted the essay.
    text   : The essay as it was originally submitted to Write & Improve as a long string.
    cefr   : A CEFR level for the text. These are more fine-grained than the M2 version of the corpus; e.g. A2.ii.
    edits  : A list of all the character level edits made to the text by all annotators, of the form:
            [[annotator_id, [[char_start_offset, char_end_offset, correction], ...]], ...].

-- M2 --
The M2 format is the processed version of the corpus that we recommend for the BEA2019 shared task.
M2 format has been the standard format for annotated GEC files since the first CoNLL shared task in 2013.
See the previous shared tasks, the ERRANT readme, or the BEA2019 shared task website for more information about M2 format.

Since it is not easy to convert character level edits in unprocessed text into token level edits in sentences (cf. https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-894.pdf), we provide a json_to_m2.py script to convert the raw JSON to M2. This script must be placed inside the main directory of the ERRor ANnotation Toolkit (ERRANT) in order to be used. ERRANT is available here: https://github.com/chrisjbryant/errant

Each M2 file was thus generated in Python 3.5 using the following command:

python3 errant/json_to_m2.py <wi_json> -out <wi_m2> -gold

This used spacy v1.9.0 and the en_core_web_sm-1.2.0 model.

Updates
----------------------

-- v2.0 --

* All punctuation was normalised in the M2 files. It was otherwise arbitrary whether, for example, different apostrophe styles were corrected or not.

* Fixed a bug in the character to token edit conversion script.

-- v2.1 --

* Updated some of the numbers in the statistics in the readme.

* Updated the json_to_m2.py script to handle multiple annotators.

* 02/07/20 - Updated the json_to_m2.py script to work with ERRANT v2.

* 02/07/20 - Added a test folder containing the tokenised W&I+LOCNESS test data. This file was previously only available on Codalab.
