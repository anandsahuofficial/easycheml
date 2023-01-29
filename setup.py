import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='easychemai',  
     version='0.1',
     scripts=['easychemai'] ,
     author="Anand Sahu",
     author_email="anandsahuofficial@gmail.com",
     description="A simple tool for using artificial intelligence in chemistry",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )