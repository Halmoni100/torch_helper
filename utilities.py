import os
import sys
import pathlib

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

def set_stdout_unbuffered():
    sys.stdout = Unbuffered(sys.stdout)

def move_state_dict(state_dict, device):
    for k, v in state_dict.items():
        state_dict[k] = v.to(device)

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
