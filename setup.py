import setuptools

__version__ = '0.0.0'

REPO_NAME = "gnn"
AUTHOR_NAME = "Tan Chia Yan"
SRC_REPO = "amogel"
AUTHOR_EMAIL = "tchiayan@gmail.com"


setuptools.setup(
    name=SRC_REPO, 
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)