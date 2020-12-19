# -*- coding: utf-8 -*-
# Copyright 2020 Unknot.id Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Various utility functions to use throughout the project."""


import copy
import functools
import heapq
import numpy as np
from tqdm.autonotebook import tqdm
import shutil
import math


def format_translation_output(sentence,
                              score=None,
                              token_level_scores=None,
                              attention=None,
                              alignment_type=None):
    """Formats a translation output with possibly scores, alignments, etc., e.g:

      1.123214 ||| Hello world ||| 0.30907777 0.030488174 ||| 0-0 1-1

    Args:
      sentence: The translation to output.
      score: If set, attach the score.
      token_level_scores: If set, attach the token level scores.
      attention: The attention vector.
      alignment_type: The type of alignments to format (can be: "hard", "soft").
    """
    if score is not None:
        sentence = "%f ||| %s" % (score, sentence)
    if token_level_scores is not None:
        scores_str = " ".join("%f" % s for s in token_level_scores)
        sentence = "%s ||| %s" % (sentence, scores_str)
    if attention is not None and alignment_type is not None:
        if alignment_type == "hard":
            source_indices = np.argmax(attention, axis=-1)
            target_indices = range(attention.shape[0])
            pairs = ("%d-%d" % (src, tgt) for src, tgt in zip(source_indices, target_indices))
            sentence = "%s ||| %s" % (sentence, " ".join(pairs))
        elif alignment_type == "soft":
            vectors = []
            for vector in attention:
                vectors.append(" ".join("%.6f" % value for value in vector))
            sentence = "%s ||| %s" % (sentence, " ; ".join(vectors))
        else:
            raise ValueError("Invalid alignment type %s" % alignment_type)
    return sentence


def item_or_tuple(x):
    """Returns :obj:`x` as a tuple or its single element."""
    x = tuple(x)
    if len(x) == 1:
        return x[0]
    else:
        return x


# def count_lines(filename, buffer_size=65536):
#     """Returns the number of lines of the file :obj:`filename`."""
#     with tf.io.gfile.GFile(filename, mode="rb") as f:
#         num_lines = 0
#         while True:
#             data = f.read(buffer_size)
#             if not data:
#                 return num_lines
#             num_lines += data.count(b"\n")


def is_gzip_file(filename):
    """Returns ``True`` if :obj:`filename` is a GZIP file."""
    return filename.endswith(".gz")


# def shape_list(x):
#     """Return list of dims, statically where possible."""
#     x = tf.convert_to_tensor(x)
#
#     # If unknown rank, return dynamic shape
#     if x.shape.dims is None:
#         return tf.shape(x)
#
#     static = x.shape.as_list()
#     shape = tf.shape(x)
#
#     ret = []
#     for i, _ in enumerate(static):
#         dim = static[i]
#         if dim is None:
#             dim = shape[i]
#         ret.append(dim)
#     return ret


def index_structure(structure, path, path_separator="/"):
    """Follows :obj:`path` in a nested structure of objects, lists, and dicts."""
    keys = path.split(path_separator)
    for i, key in enumerate(keys):
        current_path = "%s%s" % (path_separator, path_separator.join(keys[:i]))
        if isinstance(structure, list):
            try:
                index = int(key)
            except ValueError:
                raise ValueError("Object referenced by path '%s' is a list, but got non "
                                 "integer index '%s'" % (current_path, key))
            if index < 0 or index >= len(structure):
                raise ValueError("List referenced by path '%s' has length %d, but got "
                                 "out of range index %d" % (current_path, len(structure), index))
            structure = structure[index]
        elif isinstance(structure, dict):
            structure = structure.get(key)
            if structure is None:
                raise ValueError("Dictionary referenced by path '%s' does not have the "
                                 "key '%s'" % (current_path, key))
        else:
            structure = getattr(structure, key, None)
            if structure is None:
                raise ValueError("Object referenced by path '%s' does not have the "
                                 "attribute '%s'" % (current_path, key))
    return structure


def clone_layer(layer):
    """Clones a layer."""
    return copy.deepcopy(layer)


def extract_batches(tensors):
    """Returns a generator to iterate on each batch of a Numpy array or dict of
  Numpy arrays."""
    if not isinstance(tensors, dict):
        for tensor in tensors:
            yield tensor
    else:
        batch_size = None
        for value in tensors.values():
            batch_size = batch_size or value.shape[0]
        for b in range(batch_size):
            yield {
                key: value[b] for key, value in tensors.items()
            }


def extract_prefixed_keys(dictionary, prefix):
    """Returns a dictionary with all keys from :obj:`dictionary` that are prefixed
  with :obj:`prefix`.
  """
    sub_dict = {}
    for key, value in dictionary.items():
        if key.startswith(prefix):
            original_key = key[len(prefix):]
            sub_dict[original_key] = value
    return sub_dict


def extract_suffixed_keys(dictionary, suffix):
    """Returns a dictionary with all keys from :obj:`dictionary` that are suffixed
  with :obj:`suffix`.
  """
    sub_dict = {}
    for key, value in dictionary.items():
        if key.endswith(suffix):
            original_key = key[:-len(suffix)]
            sub_dict[original_key] = value
    return sub_dict


def merge_dict(dict1, dict2):
    """Merges :obj:`dict2` into :obj:`dict1`.

  Args:
    dict1: The base dictionary.
    dict2: The dictionary to merge.

  Returns:
    The merged dictionary :obj:`dict1`.
  """
    for key, value in dict2.items():
        if isinstance(value, dict):
            dict1[key] = merge_dict(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1


class OrderRestorer(object):
    """Helper class to restore out-of-order elements in order."""

    def __init__(self, index_fn, callback_fn):
        """Initializes this object.

    Args:
      index_fn: A callable mapping an element to a unique index.
      callback_fn: A callable taking an element that will be called in order.
    """
        self._index_fn = index_fn
        self._callback_fn = callback_fn
        self._next_index = 0
        self._elements = {}
        self._heap = []

    @property
    def buffer_size(self):
        """Number of elements waiting to be notified."""
        return len(self._heap)

    @property
    def next_index(self):
        """The next index to be notified."""
        return self._next_index

    def _try_notify(self):
        old_index = self._next_index
        while self._heap and self._heap[0] == self._next_index:
            index = heapq.heappop(self._heap)
            value = self._elements.pop(index)
            self._callback_fn(value)
            self._next_index += 1
        return self._next_index != old_index

    def push(self, x):
        """Push event :obj:`x`."""
        index = self._index_fn(x)
        if index is None:
            self._callback_fn(x)
            return True
        if index < self._next_index:
            raise ValueError("Event index %d was already notified" % index)
        self._elements[index] = x
        heapq.heappush(self._heap, index)
        return self._try_notify()


class ClassRegistry(object):
    """Helper class to create a registry of classes."""

    def __init__(self, base_class=None):
        """Initializes the class registry.

    Args:
      base_class: Ensure that classes added to this registry are a subclass of
        :obj:`base_class`.
    """
        self._base_class = base_class
        self._registry = {}

    @property
    def class_names(self):
        """Class names registered in this registry."""
        return set(self._registry.keys())

    def register(self, cls=None, name=None, alias=None):
        """Registers a class.

    Args:
      cls: The class to register. If not set, this method returns a decorator for
        registration.
      name: The class name. Defaults to ``cls.__name__``.
      alias: An optional alias or list of alias for this class.

    Returns:
      :obj:`cls` if set, else a class decorator.

    Raises:
      TypeError: if :obj:`cls` does not extend the expected base class.
      ValueError: if the class name is already registered.
    """
        if cls is None:
            return functools.partial(self.register, name=name, alias=alias)
        if self._base_class is not None and not issubclass(cls, self._base_class):
            raise TypeError("Class %s does not extend %s"
                            % (cls.__name__, self._base_class.__name__))
        if name is None:
            name = cls.__name__
        self._register(cls, name)
        if alias is not None:
            if not isinstance(alias, (list, tuple)):
                alias = (alias,)
            for alias_name in alias:
                self._register(cls, alias_name)
        return cls

    def _register(self, cls, name):
        if name in self._registry:
            raise ValueError("Class name %s is already registered" % name)
        self._registry[name] = cls

    def get(self, name):
        """Returns the class with name :obj:`name` or ``None`` if it does not exist
    in the registry.
    """
        return self._registry.get(name)


def get_terminal_width():
    width = shutil.get_terminal_size(fallback=(200, 24))[0]
    if width == 0:
        width = 120
    return width


def pbar(total_records, batch_size, epoch, epochs):
    bar = tqdm(total=math.ceil(total_records / batch_size) * batch_size,
               ncols=int(get_terminal_width() * .9),
               # desc=tqdm.write(f'Epoch {epoch + 1}/{epochs}'),
               postfix={
                   'g_loss': f'{0:6.3f}',
                   'd_loss': f'{0:6.3f}',
                   'epoch': epoch + 1,
                   'epochs': epochs,
                   1: 1
               },
               bar_format='Epoch {postfix[epoch]}/{postfix[epochs]} | '
                          '{n_fmt}/{total_fmt} | {bar} | {rate_fmt}  '
               'ETA: {remaining}  Elapsed Time: {elapsed}  '
               'D Loss: {postfix[d_loss]}  G Loss: {postfix[g_loss]}',
               unit=' record',
               miniters=10)
    return bar
