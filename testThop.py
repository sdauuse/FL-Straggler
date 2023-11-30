# author: miaoyin
# time: 2023-11-13 12:24
import time

from numpy.random import randint

import utils
from models import FC2Layer
from utils import write_list_to_csv
from configurations import args_parser

data = [1, 2, 3]
stragglers_percent = 0.5

stragglers_idx = randint(low=0, high=30, size=round(0.9 * 30))
i = randint(low=0, high=30, size=round(stragglers_percent * 100))
print(i)
