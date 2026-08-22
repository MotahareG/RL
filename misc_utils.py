# misc_utils.py  –  Fixed version
# Fixes:
#   1. print_time: `(time.time()-start) + "  " + str(...)` crashes (float + str).
#      Fixed to use string formatting.
#   2. to_vars: used torch.cuda / Variable without importing torch.
#      Guarded with try/except and note about dependency.

from __future__ import print_function
import json, math, os, sys, time
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO

print_grad = True


class printOut(object):
    def __init__(self, f=None, stdout_print=True):
        self.out_file     = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        if self.out_file:
            try:
                self.out_file.write(s)
            except UnicodeEncodeError:
                self.out_file.write(s.encode('ascii', 'replace').decode('ascii'))
            if new_line:
                self.out_file.write("\n")
            self.out_file.flush()
        if self.stdout_print:
            try:
                print(s, end="", file=sys.stdout)
            except UnicodeEncodeError:
                print(s.encode('ascii', 'replace').decode('ascii'), end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self, s, start_time):
        """Print elapsed time and return a new timestamp."""
        # FIX: was `(time.time() - start_time) + "  " + str(...)` → TypeError
        elapsed = time.time() - start_time
        self.print_out("{}, time {:.0f}s, {}.".format(s, elapsed, time.ctime()))
        return time.time()

    def print_grad(self, model, last=False):
        if print_grad:
            for tag, value in model.named_parameters():
                if value.grad is not None:
                    self.print_out(
                        '{:<50}\t-- value: {:.12f}\t-- grad: {}'.format(
                            tag, value.norm().data[0], value.grad.norm().data[0]))
                else:
                    self.print_out(
                        '{:<50}\t-- value: {:.12f}'.format(tag, value.norm().data[0]))
            self.print_out("-" * 35)
            if last:
                self.print_out("-" * 35)
                self.print_out("-" * 35)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def to_np(x):
    return x.data.cpu().numpy()


def to_vars(x):
    """Move tensor to GPU if available (requires PyTorch)."""
    # FIX: original code used torch without importing it
    try:
        import torch
        from torch.autograd import Variable
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    except ImportError:
        raise ImportError("PyTorch is required for to_vars(). "
                          "This project primarily uses TensorFlow.")


def extract(xVar):
    global yGrad
    yGrad = xVar
    print(yGrad)


def extract_norm(xVar):
    global yGrad
    yGradNorm = xVar.norm()
    print(yGradNorm)


class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        bin_edges = bin_edges[1:]
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    cfg = tf.ConfigProto(log_device_placement=log_device_placement,
                         allow_soft_placement=allow_soft_placement)
    cfg.gpu_options.allow_growth = True
    return cfg


def gradient_clip(gradients, params, max_gradient_norm):
    clipped, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    summary = [tf.summary.scalar("grad_norm", norm),
               tf.summary.scalar("clipped_gradient", tf.global_norm(clipped))]
    return clipped, summary


def add_summary(summary_writer, global_step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def get_device_str(device_id, num_gpus):
    if num_gpus == 0:
        return "/cpu:0"
    return "/gpu:%d" % (device_id % num_gpus)


def debug_tensor(s, msg=None, summarize=10):
    if not msg:
        msg = s.name
    return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)