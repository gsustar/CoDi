"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import os.path as osp
import json
import time
import datetime
import tempfile
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
from .plot_util import to_pil_image


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class KVWriter(object):
	def writekvs(self, kvs):
		raise NotImplementedError


class SeqWriter(object):
	def writeseq(self, seq):
		raise NotImplementedError


class ImgWriter(object):
	def writeimg(self, img):
		raise NotImplementedError
	

class GifWriter(object):
	def writegif(self, gif):
		raise NotImplementedError


class TensorWriter(object):
	def writetensor(self, tensor):
		raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter, ImgWriter, GifWriter, TensorWriter):
	def __init__(self, filename_or_file):
		if isinstance(filename_or_file, str):
			self.file = open(filename_or_file, "wt")
			self.own_file = True
		else:
			assert hasattr(filename_or_file, "read"), (
				"expected file or str, got %s" % filename_or_file
			)
			self.file = filename_or_file
			self.own_file = False
		
		self.filename_or_file = filename_or_file
		self.mediadir = None
		# if filename_or_file != sys.stdout:
		# 	self.mediadir = osp.join(osp.dirname(filename_or_file), "media")
		# 	os.makedirs(self.mediadir, exist_ok=True)

	def writekvs(self, kvs):
		# Create strings for printing
		key2str = {}
		for (key, val) in sorted(kvs.items()):
			if hasattr(val, "__float__"):
				valstr = "%-8.3g" % val
			else:
				valstr = str(val)
			key2str[self._truncate(key)] = self._truncate(valstr)

		# Find max widths
		if len(key2str) == 0:
			print("WARNING: tried to write empty key-value dict")
			return
		else:
			keywidth = max(map(len, key2str.keys()))
			valwidth = max(map(len, key2str.values()))

		# Write out the data
		dashes = "-" * (keywidth + valwidth + 7)
		lines = [dashes]
		for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
			lines.append(
				"| %s%s | %s%s |"
				% (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
			)
		lines.append(dashes)
		self.file.write("\n".join(lines) + "\n")

		# Flush the output to the file
		self.file.flush()

	def _truncate(self, s):
		maxlen = 30
		return s[: maxlen - 3] + "..." if len(s) > maxlen else s

	def writeseq(self, seq):
		seq = list(seq)
		for (i, elem) in enumerate(seq):
			self.file.write(elem)
			if i < len(seq) - 1:  # add space unless this is the last one
				self.file.write(" ")
		self.file.write("\n")
		self.file.flush()

	def _get_media_savedir(self, step):
		savedir = self.mediadir
		if step is not None:
			subdir = f"step_{step}" if isinstance(step, int) else step
			savedir = osp.join(
				self.mediadir, subdir
			)
		os.makedirs(savedir, exist_ok=True)
		return savedir

	def writeimg(self, tensor, name, step=None, **grid_kwargs):
		if self.mediadir is None:
			return
		savedir = self._get_media_savedir(step)
		img = to_pil_image(tensor, **grid_kwargs)
		img.save(
			osp.join(savedir, f"{name}.png")
		)
	
	def writegif(self, lst_of_tensors, name, step=None):
		if self.mediadir is None:
			return
		savedir = self._get_media_savedir(step)
		frames = [to_pil_image(t) for t in lst_of_tensors]
		frames[0].save(osp.join(savedir, f"{name}.gif"), save_all=True, 
				 	append_images=frames[1:], duration=40, loop=0)

	def writetensor(self, tensor, name, step=None):
		if self.mediadir is None:
			return
		savedir = self._get_media_savedir(step)
		with open(os.path.join(savedir, f"{name}.npy"), "wb") as fout:
			np.save(fout, tensor.contiguous().clone().cpu().numpy())

	def close(self):
		if self.own_file:
			self.file.close()


class JSONOutputFormat(KVWriter):
	def __init__(self, filename):
		self.file = open(filename, "wt")

	def writekvs(self, kvs):
		for k, v in sorted(kvs.items()):
			if hasattr(v, "dtype"):
				kvs[k] = float(v)
		self.file.write(json.dumps(kvs) + "\n")
		self.file.flush()

	def close(self):
		self.file.close()


class CSVOutputFormat(KVWriter):
	def __init__(self, filename):
		self.file = open(filename, "w+t")
		self.keys = []
		self.sep = ","

	def writekvs(self, kvs):
		# Add our current row to the history
		extra_keys = list(kvs.keys() - self.keys)
		extra_keys.sort()
		if extra_keys:
			self.keys.extend(extra_keys)
			self.file.seek(0)
			lines = self.file.readlines()
			self.file.seek(0)
			for (i, k) in enumerate(self.keys):
				if i > 0:
					self.file.write(",")
				self.file.write(k)
			self.file.write("\n")
			for line in lines[1:]:
				self.file.write(line[:-1])
				self.file.write(self.sep * len(extra_keys))
				self.file.write("\n")
		for (i, k) in enumerate(self.keys):
			if i > 0:
				self.file.write(",")
			v = kvs.get(k)
			if v is not None:
				self.file.write(str(v))
		self.file.write("\n")
		self.file.flush()

	def close(self):
		self.file.close()


class WandbOutputFormat(KVWriter, ImgWriter, GifWriter):
	"""
	Log using `Weights and Biases`.
	"""
	def __init__(self, dir, wandb_kwargs):
		try:
			import wandb
		except ImportError:
			raise ImportError(
				"To use the Weights and Biases Logger please install wandb."
				"Run `pip install wandb` to install it."
			)
		
		self._wandb = wandb

		# Initialize a W&B run
		if self._wandb.run is None:
			self._wandb.init(
				project=wandb_kwargs.get("project", None),
				dir=dir,
				group=wandb_kwargs.get("group", None),
				config=wandb_kwargs.get("config", None),
				name=wandb_kwargs.get("name", None),
				mode=wandb_kwargs.get("mode", "online"),
			)

	def writekvs(self, kvs):
		step = kvs.pop("step")
		# self._wandb.log(kvs, commit=True, step=step)
		self._wandb.log(kvs, commit=True)

	def writeimg(self, tensor, name, step=None):
		img = to_pil_image(tensor)
		self._wandb.log(
			{f"media/{name}": [self._wandb.Image(img)]},
			commit=True
		)
	
	def writegif(self, lst_of_tensors, name, step=None):
		frames = np.array(
			[to_pil_image(t) for t in lst_of_tensors],
			dtype=np.uint8
		).transpose(0, 3, 1, 2)
		self._wandb.log(
			{f"media/{name}": self._wandb.Video(frames, fps=24)},
			commit=True
		)
	
	def close(self):
		self._wandb.finish()


def make_output_format(format, ev_dir, log_suffix="", wandb_kwargs=dict()):
	if format == "stdout":
		return HumanOutputFormat(sys.stdout)
	elif format == "log":
		return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
	elif format == "json":
		return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
	elif format == "csv":
		return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
	elif format == "wandb":
		return WandbOutputFormat(ev_dir, wandb_kwargs)
	else:
		raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
	"""
	Log a value of some diagnostic
	Call this once for each diagnostic quantity, each iteration
	If called many times, last value will be used.
	"""
	get_current().logkv(key, val)


def logkv_mean(key, val):
	"""
	The same as logkv(), but if called many times, values averaged.
	"""
	get_current().logkv_mean(key, val)


def logkvs(d):
	"""
	Log a dictionary of key-value pairs
	"""
	for (k, v) in d.items():
		logkv(k, v)


def logimg(tensor, name, step=None, **kwargs):
	"""
	Log an image
	"""
	get_current().logimg(tensor, name, step, **kwargs)


def loggif(lst_of_tensors, name, step=None, **kwargs):
	"""
	Log a gif
	"""
	get_current().loggif(lst_of_tensors, name, step, **kwargs)


def savetensor(tensor, name, step=None):
	"""
	Save a tensor as a numpy array
	"""
	get_current().savetensor(tensor, name, step)


def dumpkvs():
	"""
	Write all of the diagnostics from the current iteration
	"""
	return get_current().dumpkvs()


def getkvs():
	return get_current().name2val


def log(*args, level=INFO):
	"""
	Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
	"""
	get_current().log(*args, level=level)


def debug(*args):
	log(*args, level=DEBUG)


def info(*args):
	log(*args, level=INFO)


def warn(*args):
	log(*args, level=WARN)


def error(*args):
	log(*args, level=ERROR)


def set_level(level):
	"""
	Set logging threshold on current logger.
	"""
	get_current().set_level(level)


def get_dir():
	"""
	Get directory that log files are being written to.
	will be None if there is no output directory (i.e., if you didn't call start)
	"""
	return get_current().get_dir()


def get_mediadir():
	return get_current().get_mediadir()


def set_mediadir(mediadir):
	_mediadir = mediadir
	counter = 1
	while osp.exists(osp.join(get_dir(), _mediadir)):
		_mediadir = mediadir + str(counter)
		counter += 1
	if mediadir != _mediadir:
		log(f"'{mediadir}' directory already exists. Using '{_mediadir}' instead...")
	get_current().set_mediadir(_mediadir)


def close():
	get_current().close()


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
	logkey = "wait_" + scopename
	tstart = time.time()
	try:
		yield
	finally:
		get_current().name2val[logkey] += time.time() - tstart


def profile(n):
	"""
	Usage:
	@profile("my_func")
	def my_func(): code
	"""

	def decorator_with_name(func):
		def func_wrapper(*args, **kwargs):
			with profile_kv(n):
				return func(*args, **kwargs)

		return func_wrapper

	return decorator_with_name


# ================================================================
# Backend
# ================================================================


def get_current():
	if Logger.CURRENT is None:
		_configure_default_logger()

	return Logger.CURRENT


class Logger(object):
	DEFAULT = None  # A logger with no output files. (See right below class definition)
	# So that you can still log to the terminal without setting up any output files
	CURRENT = None  # Current logger being used by the free functions above

	def __init__(self, dir, output_formats):
		self.name2val = defaultdict(float)  # values this iteration
		self.name2cnt = defaultdict(int)
		self.level = INFO
		self.dir = dir
		self.output_formats = output_formats

	# Logging API, forwarded
	# ----------------------------------------
	def logkv(self, key, val):
		self.name2val[key] = val

	def logkv_mean(self, key, val):
		oldval, cnt = self.name2val[key], self.name2cnt[key]
		self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
		self.name2cnt[key] = cnt + 1

	def dumpkvs(self):
		for fmt in self.output_formats:
			if isinstance(fmt, KVWriter):
				fmt.writekvs(self.name2val)
		self.name2val.clear()
		self.name2cnt.clear()

	def log(self, *args, level=INFO):
		if self.level <= level:
			self._do_log(args)

	def logimg(self, tensor, name, step=None, **kwargs):
		for fmt in self.output_formats:
			if isinstance(fmt, ImgWriter):
				fmt.writeimg(tensor, name, step, **kwargs)
	
	def loggif(self, lst_of_tensors, name, step=None, **kwargs):
		for fmt in self.output_formats:
			if isinstance(fmt, GifWriter):
				fmt.writegif(lst_of_tensors, name, step, **kwargs)

	def savetensor(self, tensor, name, step=None):
		for fmt in self.output_formats:
			if isinstance(fmt, TensorWriter):
				fmt.writetensor(tensor, name, step)

	# Configuration
	# ----------------------------------------
	def set_level(self, level):
		self.level = level

	def get_dir(self):
		return self.dir
	
	def get_mediadir(self):
		for fmt in self.output_formats:
			if isinstance(fmt, HumanOutputFormat):
				if fmt.mediadir is not None:
					return fmt.mediadir
		return None
	
	def set_mediadir(self, mediadir):
		for fmt in self.output_formats:
			if isinstance(fmt, HumanOutputFormat):
				if fmt.filename_or_file is not sys.stdout:
					mediadir = osp.join(self.dir, mediadir)
					os.makedirs(mediadir, exist_ok=True)
					fmt.mediadir = mediadir

	def close(self):
		for fmt in self.output_formats:
			fmt.close()

	# Misc
	# ----------------------------------------
	def _do_log(self, args):
		for fmt in self.output_formats:
			if isinstance(fmt, SeqWriter):
				fmt.writeseq(map(str, args))


def configure(dir=None, format_strs=None, log_suffix="", wandb_kwargs=dict(), mediadir="media"):
	if not dir:
		dir = osp.join(
			tempfile.gettempdir(),
			datetime.datetime.now().strftime("diffcount-%Y-%m-%d-%H-%M-%S-%f"),
		)
	assert isinstance(dir, str)
	dir = os.path.expanduser(dir)
	os.makedirs(dir, exist_ok=True)

	if format_strs is None:
		# format_strs = "stdout,log,csv".split(",")
		format_strs = ["stdout"]
	format_strs = filter(None, format_strs)
	output_formats = [make_output_format(f, dir, log_suffix, wandb_kwargs) for f in format_strs]

	Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
	Logger.CURRENT.set_mediadir(mediadir)
	if output_formats:
		log("Logging to %s" % dir)


def _configure_default_logger():
	configure()
	Logger.DEFAULT = Logger.CURRENT


def reset():
	if Logger.CURRENT is not Logger.DEFAULT:
		Logger.CURRENT.close()
		Logger.CURRENT = Logger.DEFAULT
		log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None):
	prevlogger = Logger.CURRENT
	configure(dir=dir, format_strs=format_strs)
	try:
		yield
	finally:
		Logger.CURRENT.close()
		Logger.CURRENT = prevlogger