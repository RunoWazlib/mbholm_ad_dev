# mbholm_ad_dev

This project is designed for the Holmstrom Lab at the University of Kansas (Follow link below).
https://www.theholmstromlab.com/

We seek to develop a machine learning algorithm that can detect Single Molecule FRET (smFRET) Bursts free from arbitrary threshold values

Our strategy is to implement anomaly detection algorithms to detect possible bursts as smFRET data largely contains background measurements

Due to the nature of smFRET Burst data, we feel that an unsupervised machine learning method is best as we ourselves are unable to classify bursts without arbitrary thresholds

Currently, we are using the IForest algorithm as our method of anomaly detection

# Requirements

Requires Python3, 3.11 preferred.

Install Python from python.org

`python -m pip install --upgrade pip`

`python -m pipenv install`

To activate a Python virtual environment created by `pipenv`:
`python -m pipenv shell`
or follow directions after the `pipenv install` command.

The `pipenv install` command creates a Pipfile and Pipfile.lock file that need to be committed to source control.

To add modules to the new Python environment:
`pipenv install <module>`
while in an active `pipenv shell` session.
