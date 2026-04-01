# make_mocks/setup.py

from setuptools import setup, find_packages

setup(
    name='make_mocks',
    version='0.1',
    description='Mock galaxy imaging pipeline and adaptive smoothing tools',
    author='Garreth Martin',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'astropy',
        'fast_histogram',
    ],
    include_package_data=True,
    package_data={
        'make_mocks': ['templates/**/*', 'filters/**/*'],
    },
    zip_safe=False,
)
