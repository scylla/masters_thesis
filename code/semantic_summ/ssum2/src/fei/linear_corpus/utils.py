#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


class Queue:

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


class Stack:

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


def getLogger(log_name='', log_file='file.log'):
    """
    get logger
    """
    # logging.getLogger() is a singleton
    logger = logging.getLogger(log_name)
    formatter = logging.Formatter(logging.BASIC_FORMAT)

    if not len(logger.handlers):

        # add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # add file handler
        file_handler_info = logging.FileHandler(log_file, mode='w')
        file_handler_info.setFormatter(formatter)
        file_handler_info.setLevel(logging.DEBUG)
        logger.addHandler(file_handler_info)

        logger.setLevel(logging.DEBUG)

    return logger
