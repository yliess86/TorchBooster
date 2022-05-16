try:
    import coloredlogs
    coloredlogs.install(fmt='%(asctime)s - %(levelname)s - %(message)s')
except ImportError:
    # fallback to default logging package
    import logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

__version__ = "0.0.3"
