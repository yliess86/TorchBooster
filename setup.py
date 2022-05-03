from distutils.core import setup
from torchbooster import __version__


setup(
    name="torchbooster",
    version=__version__,
    description="A simple and yet practical pytorch booster library to help with bootstraping and reproducing research.",
    author="Yliess Hati and Gregor Jouet",
    author_email="yliess.hati@devinci.fr",
    packages=["torchbooster"],
    package_dir={"torchbooster": "torchbooster"},
    keywords=["research", "deep-learning", "reproducible-research", "python3", "pytorch"],
    license="MIT",
    extras_require = {'huggingface_datasets':  ['datasets'], 'colored_logs': ['coloredlogs'] }
)
