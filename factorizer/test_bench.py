# Created by ay27 at 16/11/8
import unittest
import logging
from logging.config import fileConfig

if __name__ == "__main__":
    fileConfig('../conf/logging_config.ini')
    # logging.getLogger().setLevel(logging.ERROR)
    suite = unittest.TestLoader().discover('.', pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(suite)
