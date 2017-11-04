"""
A setuptools-based setup module.
"""

from setuptools import setup, find_packages

setup(
    name='hopfield',
    version='0.0.1',
    description='Train and use Hopfield networks.',
    long_description='Train and use Hopfield networks.',
    url='https://github.com/unixpickle/hopfield',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ai hopfield neural network',
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'numpy>=1.0.0,<2.0.0'
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
    }
)
