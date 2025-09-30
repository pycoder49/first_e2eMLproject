from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    '''
    Reads a requirements file and returns a list of dependencies.
    '''
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

 
setup(
    name = "e2eML",
    version = "0.1.0",
    author = "Aryan Ahuja",
    author_email = "aryan-a@outlook.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)