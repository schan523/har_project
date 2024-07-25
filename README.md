Human Activity Recognition Task
==============================

Training and deploying a CNN on a dataset of human activity images.

All preprocessing, training, and predicting is done n the Jupyter notebook in the notebooks directory.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

<p>To clone this repository, run:
    
    git clone https://github.com/schan523/har_project.git

</p>

## Contributing

### Overview of Terms

1. **Forking or Cloning**: A developer starts by creating a copy (fork) of the central repository to their own account or by cloning the repository to their local machine.
2. **Creating a Branch**: The developer creates a new branch (usually a feature branch) in their local copy of the repository. This branch will contain the changes they want to propose.
3. **Making Changes**: The developer makes changes to the code, adds new features, fixes bugs, or performs other modifications in their feature branch. They commit these changes locally.
4. **Pushing the Branch**: After making the desired changes, the developer pushes their feature branch to the central repository (usually to their own fork or a branch in the same repository).
5. **Creating a Pull Request**: The developer then opens a pull request in the central repository. This is a formal request to merge the changes from their feature branch into a target branch, such as `main` or `develop`. They describe the purpose of the changes, provide context, and include any relevant information.
6. **Code Review**: Team members and collaborators review the pull request. They can comment on the code, suggest improvements, and ensure that it adheres to coding standards and project guidelines.
7. **Continuous Integration (CI)/Continuous Deployment (CD)**: Automated CI/CD processes can be triggered to build and test the changes to ensure that they don't introduce errors or break existing functionality.
8. **Discussion and Iteration**: The team and the developer can engage in discussions within the pull request, addressing any questions, concerns, or suggestions. The developer can make additional commits to their feature branch to address feedback.
9. **Merge**: Once the pull request is reviewed and approved, a team member with merge permissions can merge the changes into the target branch. This integrates the proposed changes into the main codebase.
10. **Closing the Pull Request**: After the changes are successfully merged, the pull request is closed. Depending on the platform, it may be marked as merged or closed, and the branch associated with the pull request can also be deleted if it's no longer needed.

### Branching Convention

We use feature branching in order to develop of the `main` branch. Once we start integrating development, we can use a `main` branch for deployment and use `development` as our development branch. 

#### Branch Naming

- `feature/<feature-name>`: Feature development branches.
- `bugfix/<bug-description>`: Bug fix branches.
- `hotfix/<hotfix-description>`: Hotfix branches for critical issues.
- `<name>`: Personal branches for playing with the code.

#### Pushing

- Do not push directly to `main` or `development` for changes to code.
- In order to push code changes to `main` or `development` you must place a pull request, and then once approved it will merge with the destination. 
- Push freely to the other branches you are working on, and you can set permsions for the branches you are working on to protect from unintended concequences.  

