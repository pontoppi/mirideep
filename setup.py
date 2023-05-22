import setuptools

setuptools.setup(
    name="mirideep",
    version="6.0",
    author="Klaus Pontoppidan",
    author_email="kpontoppi@gmail.com",
    description="A package to calibrate high signal-to-noise MIRI MRS data",
    packages=['mirideep'],
    package_data={'mirideep': ['rsrfs/*.npz','rsrfs/*.fits','rsrfs/*.csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
