from setuptools import find_packages,setup
from typing import List

PROJECT_NAME = 'Big Mart Store Sales Prediction'
VERSION = '0.0.1'
AUTHOR = 'Vijaykumar Rewate'
AUTHOR_EMAIL = 'rewatevijaykumar@gmail.com'
DESCRIPTION = 'Big Mart Store Sales Prediction using Machine Learning'
HYPEN_E_DOT = '-e .'
REQUIREMENTS_FILE = 'requirements.txt'

def get_requirements(file_path:str)-> List[str]:
    '''
    Returns a list of requirement packages mentioned in the requirements.txt file
    '''
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements.append(file_obj.readline())
    requirements = [requirement.replace("/n",'') for requirement in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirements(REQUIREMENTS_FILE),
)

