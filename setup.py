import setuptools

setuptools.setup(
	name="fcassim",
    version="0.0.1",
    author="Columbia NLP Lab",
    author_email="xy2437@columbia.edu",
    description="Cassim with LTK Support",
    url="https://github.com/jasonyux/FastCASSIM.git",
    package_dir={"": "fcassim"},
    packages=setuptools.find_packages("fcassim"),
    python_requires=">=3.6",
    test_suite='nose.collector',
    tests_require=['nose'],
)