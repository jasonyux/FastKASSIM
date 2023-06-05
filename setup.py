import setuptools

setuptools.setup(
    name="fkassim",
    version="0.0.1",
    author="Columbia NLP Lab",
    author_email="xy2437@columbia.edu",
    description="kassim with LTK Support",
    install_requires=["nltk", "numpy", "scipy", "zss"],
    packages=setuptools.find_packages(),
    py_modules=["fkassim"],
    python_requires=">=3.6",
    test_suite='nose.collector',
    tests_require=['nose'],
    url="https://github.com/jasonyux/FastKassim.git",
)
