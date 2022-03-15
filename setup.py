import setuptools

setuptools.setup(
	name="fkassim",
    version="0.0.1",
    author="Columbia NLP Lab",
    author_email="xy2437@columbia.edu",
    description="kassim with LTK Support",
    url="https://github.com/jasonyux/FastKassim.git",
    package_dir={"": "fkassim"},
    packages=setuptools.find_packages("fkassim"),
    python_requires=">=3.6",
    test_suite='nose.collector',
    tests_require=['nose'],
)