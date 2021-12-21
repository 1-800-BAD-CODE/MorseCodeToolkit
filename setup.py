
from typing import List
from setuptools import setup, find_packages

install_requires: List[str]
with open("requirements.txt") as f:
    install_requires = [x.strip() for x in f]

setup(
    name="morsecodetoolkit",
    version="0.1",
    author="1-800-BAD-CODE",
    author_email="",
    description="Toolkit for morse code generation and morse code recognition model training.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "mct-text-to-morse = morsecodetoolkit.bin.text_to_morse:main",
            "mct-morse-to-text = morsecodetoolkit.bin.morse_to_text:main"
        ]
    }
)
