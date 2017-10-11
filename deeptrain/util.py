"""
The MIT License (MIT)

Copyright (c) 2017 Erik Perillo <erik.perillo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import os
import sys
import datetime
import subprocess as sp
import numpy as np

def git_hash():
    """
    Gets git commit hash of project.
    """
    try:
        hsh = sp.getoutput("git rev-parse HEAD").strip("\n")
    except:
        hsh = ""

    return hsh

def time_str():
    """
    Returns string-formatted local time in format hours:minutes:seconds.
    """
    return "".join(str(datetime.datetime.now().time()).split(".")[0])

def date_str():
    """
    Returns string-formatted local date in format year-month-day.
    """
    return str(datetime.datetime.now().date())

def uniq_filename(dir_path, pattern, ext=""):
    """
    Returns an unique filename in directory path that starts with pattern.
    """
    dir_path = os.path.abspath(dir_path)

    files = [f for f in os.listdir(dir_path) if \
             os.path.exists(os.path.join(dir_path, f))]
    num = len([f for f in files if f.startswith(pattern)])

    filename = pattern + (("-" + str(num)) if num else "") + ext

    return filename

def uniq_filepath(dir_path, pattern, ext=""):
    """
    Returns an unique filepath in directory path that starts with pattern.
    """
    dir_path = os.path.abspath(dir_path)
    return os.path.join(dir_path, uniq_filename(dir_path, pattern, ext))

def get_ext(filepath, sep="."):
    """
    Gets extension of file given a filepath.
    """
    filename = os.path.basename(filepath.rstrip("/"))
    if not sep in filename:
        return ""
    return filename.split(sep)[-1]

class Tee:
    """
    Broadcasts print message through list of open files.
    """
    def __init__(self, files):
        self.files = files

    def print(self, *args, **kwargs):
        for f in self.files:
            kwargs["file"] = f
            print(*args, **kwargs)

    def __del__(self):
        for f in self.files:
            if f != sys.stdout and f != sys.stderr:
                f.close()
