#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import setup, find_packages

setup(
    author="Cory Johns",
    author_email="cory.johns@canonical.com",
    description="Fixtures for Operators",
    long_description=Path("README.md").read_text(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Framework :: Pytest",
        "Programming Language :: Python",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    license="MIT license",
    include_package_data=True,
    keywords=["pytest", "py.test", "operators", "yaml", "asyncio"],
    name="pytest-operator",
    packages=find_packages(include=["pytest_operator"]),
    url="https://github.com/charmed-kubernetes/pytest-operator",
    version="0.11.0",
    zip_safe=True,
    install_requires=[
        "ipdb",
        "pytest",
        "pytest-asyncio",
        "pyyaml",
        "juju",
        "jinja2",
    ],
    entry_points={
        "pytest11": [
            "pytest-operator = pytest_operator.plugin",
        ]
    },
)
