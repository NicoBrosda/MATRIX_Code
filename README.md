# MATRIX_Code

## Purpose
This is a MATRIX project repository intended for easier sharing of code connected to evaluation and measurement routines.

## General Information
The code is partially commented, below are some information that might help in using the code. The code might refer to
local paths, because the measurement data is not part of this project. Thus, when using the code, it is necessary to 
modify paths to the suited path on the User's PC.
This project needs to be executed from the MATRIX_code folder level, otherwise referencing inbetween different code building blocks might not work.

#### General Troubleshooting: 
- There might occur a problem if there is not a viable LaTeX distribution on the computer. If such an Error follows any attempts to plot data after importing the file Plot_Methods.plot_standards, move to this file and set on the top the parameter use_LaTeX to False. Note that without LaTeX rendering formula are not correctly displayed in any plot.
- Check if any Error is connected to paths not adapted to your computer system.
- If any imports in this project are not working, note the info from above: The project needs to be executed from the MATRIX_code folder level, otherwise referencing inbetween different code building blocks might not work.

## Environment Setup
This project uses a conda environment to manage dependencies. Follow these steps to set up the environment:

### Prerequisites
- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download) installed
- Git installed (for cloning the repository)

### Setup Steps

1. Clone the repository:
```bash
git clone https://gitlab.ruhr-uni-bochum.de/brosdnkl/matrix_code.git
cd matrix_code
```

2. Create and activate the conda environment:
```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate MATRIX_Code
```

3. Verify the installation:
```bash
# Check if Python is using the correct environment
which python
# Should point to a path in your conda environment

# Test basic imports
python -c "import numpy; import matplotlib; import cv2; print('Environment setup successful!')"
```

### Environment Management

- To update the environment if changes are made to environment.yml:
```bash
conda env update -f environment.yml
```

- To remove the environment:
```bash
conda deactivate
conda env remove -n MATRIX_Code
```

### Troubleshooting

- If you encounter package conflicts during installation:
  - Try removing the existing environment and creating it fresh
  - Check if all required channels are properly configured in environment.yml
  - Ensure you're using the latest version of conda

- If you get permission errors:
  - Make sure you have write permissions in the installation directory
  - Try running the commands with administrator privileges if needed

- If you encounter Python version issues:
  - The environment is configured for Python 3.12
  - Make sure your conda installation is up to date
  - Check if your system supports the required Python version

## Overview
On project level only folders, the README and the license should be stored.
#### AMS_Evaluation
Contains the code for the readout and plotting of the first two measurement sets. The code is barely commented and its 
functionality has been replaced by EvaluationSoftware. read_MATRIX.py includes the relevant readout functions, different 
parts of the analysis have a separate file beginning with Console (first measurement set) or Console2 (second set).

#### Evaluation Software
Newer version of the readout, written to adapt to new array geometries (also 2D) and other readout circuits (meaning 
different file formats). Central piece is the class Analyzer containing the functionality for add measurement files and 
construct a map out of them. The readout of files, parsing of a position in the file name, possible filtering or 
correcting, etc. are all separate modules. Idea is that choosing the fitting set of modules allows to adapt the 
image generation routine to future changes.

#### Live View
Script and test for a simple LiveView window during the measurements. Using the functionality from EvaluationSoftware 
allows for creating a live updated (~5-10s for generating the first image, < 1-2s for image updating) image from a 
measurement. Requirement is, that the measurement files are stored in the folder under a unique save name (both to be 
specified in LiveView) - and that the suited array geometry with fitting modules are selected.
The file MeasurementSimulation can be used for testing purposes. It creates (under specified parameters) measurement 
files in a specified folder ('./TestMap/'). Note that the code clears all files in the folder before adding a new set, 
be careful if you change the folder.

#### Meeting_June_2024 
Should contain all presentations from the project meeting in Lacassagne June 2024.

#### Plot_Methods
Code with some matplotlib plotting functionality. Note the parameters save_format = '.png', use_LaTeX = True from 
plot_standards and language_english = True from label standard.

#### Results_Cyrcé 
Includes presentations about results of Cyrcé measurements from June 2024. 

## Maintainer
This repo was initialized by Nico Brosda, Ruhr Universität Bochum. For any questions, remarks, etc. contact nico.brosda@rub.de. Developers with permission to edit the code must be named by the project's maintainer.


## Git Start Infos

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.ruhr-uni-bochum.de/brosdnkl/matrix_code.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.ruhr-uni-bochum.de/brosdnkl/matrix_code/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
