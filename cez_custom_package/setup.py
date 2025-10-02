# /CEZ_Hackathon/cez_custom_package/setup.py

from setuptools import setup, find_packages

setup(
    name="cez-hackathon-utils",
    version="0.1.0",
    description="Custom utility functions for the CEZ Hackathon.",
    author="CEZ Hackathon Team",
    packages=find_packages(), # This will automatically find the 'utils' package
)
