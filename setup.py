
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="automeans",
    version="1.0.0",
    author="Anshul Patel",
    author_email="anshulp2912@gmail.com",
    description="A Python package that helps automate the number of cluster for k-means",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy','scikit-learn','pandas','kneed>=0.7.0', 'matplotlib>=3.3.2'],
    license = "MIT",
    url="https://github.com/anshulp2912/automeans",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 
