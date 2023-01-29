import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='easycheml',  
     version='0.1',
     author="Anand Sahu",
     author_email="anandsahuofficial@gmail.com",
     description="A simple tool for using artificial intelligence in chemistry",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/anandsahuofficial/easycheml.git",
     download_url="https://github.com/anandsahuofficial/easycheml/archive/refs/tags/easycheml.tar.gz",
     #packages=setuptools.find_packages(),
     packages=['easycheml'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )







