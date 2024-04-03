from setuptools import setup, find_packages

setup(
    name='yucl_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'langchain-core',
        'openai',
    ],
    description='A utility package for local embeddings client',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yucl80/utils',
)