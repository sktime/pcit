from distutils.core import setup
from setuptools import find_packages

setup(
    name='pcit',

    version='1.2.2',

    description='Predictive Conditional Independence Testing',

    # The project's main homepage.
    url='https://github.com/SamBurkart/pcit',

    # Author details
    author='Samuel Burkart',
    author_email='samuel.burkart@aol.com',

    # Choose your license
    license='MIT',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests'])
)