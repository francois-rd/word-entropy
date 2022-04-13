# Modeling the Behaviour of New Words with Entropy

This is the code repository for my CSC2552 course project at the University of Toronto.

## Installation

Install the required libraries listed in `requirements.txt`.

Install the following external resources:
+ The `words` corpus from NLTK: http://www.nltk.org/nltk_data/
  ~~~
  >>> import nltk
  >>> nltk.download()
  ~~~
+ The `en_core_web_sm` model from SpaCy: https://spacy.io/models/en
  ~~~
  python -m spacy download en_core_web_sm
  ~~~

Add `word-entropy/src` to `PYTHONPATH`.

Requires `Python 3.6+`.

## Replicating the main results

This project runs *way too slowly* to be done all in one session. Instead, individual steps are executed as subcommands of the `src/main.py` program. Each step is independently configurable using a JSON-style config file. Taken together, the steps form a pipeline from data download all the way to final result visualization. Running:

~~~
python src/main.py -h
~~~

produces a list of each pipeline step (in order) as well as a brief help message. Some steps rely on the output of one or more of the previous steps. To execute a step, run:

~~~
python src/main.py <step_name> --config experiments/main/pipeline/<step_name>.json
~~~

where `<step_name>` is one of the valid steps. Some steps take a very long time to run. In particular, `download` and `preprocess` take *days or weeks* to run. Therefore, to simply replicate the figures and tables from the final report, I advise running only the last three steps of the pipeline:

~~~
python src/main.py plot-ts --config experiments/main/pipeline/plot-ts.json
python src/main.py plot-stats --config experiments/main/pipeline/plot-stats.json
python src/main.py predict --config experiments/main/pipeline/predict.json
~~~

These three steps only rely on the output of the `basic-detect` and `time-series` steps. The output of these steps is provided in `experiments/main/data/neologisms` and `experiments/main/model/time_series`, respectively.

These three steps only takes a few minutes to run and produces output files in `experiments/main/results`, some of which were used in the final report. Feel free to explore the plots that didn't make the cut to gain a richer understanding of the main results.

## Additional notes

+ Contact me directly to access any of the other intermediate output files. Many of these exceed the GitHub file size limit and can therefore not be stored here.
+ Running `python src/ppt.py` produces some plots for the PowerPoint presentation. This can be ignored.
+ All other details are in the final report.
