import os
import psutil


def ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def print_ram_usage(msg=''):
    if msg:
        msg = ' (%s)' % msg
    print('RAM: %ikB%s' % (ram_usage()//1000, msg))