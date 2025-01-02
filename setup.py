from setuptools import find_packages, setup
from typing import List

def get_requirements()->List:
    """
    This function will give list of requirements
    """
    requirement_list:List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readline()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt File not found")
        
    return requirement_list


setup(
    name="Cardiac arrest predicter",
    version="0.0.1,",
    author="Arnav Lahane, Pratham, patharker,Maitreyee Deshmukh, Prachi Satpute",
    author_email="lahanearnav9@gmail.com",
    packages=find_packages(),
    install_require=get_requirements()
)