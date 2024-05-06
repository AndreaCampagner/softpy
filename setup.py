from setuptools import setup, find_packages

setup(
   name='softpy',
   version='0.0.4',
   author='Andrea Campagner',
   python_requires=">3.8.0",
   author_email='onyris93@gmail.com',
   packages=find_packages(include=['softpy', 'softpy.*']),
   url='https://pypi.org/project/scikit-weak/',
   license='LICENSE.txt',
   description='A package for soft computing in Python.',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   install_requires=[
       "numpy>=1.24.0",
       "scipy>=1.11.0",

   ],
)