#!/usr/bin/env python3

import time

from progress_bar import ProgressBar

def test_length(length):
    pbar = ProgressBar(length, length=10)
    pbar.start()
    for _ in range(length):
        time.sleep(0.1)
        pbar.update()
    time.sleep(0.1)
    pbar.stop()
    print("Finished %d!" % length)

test_length(10)