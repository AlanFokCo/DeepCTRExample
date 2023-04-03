import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import parsing_ops
import model
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import (
    SecondOrStepTimer,
)

epoch = 10
batch_size = 32


class Property(object):
    def __init__(self, name, property_name, property_doc, default=None):
        self._name = name
        self._names = property_name
        self._default = default
        self._doc = property_doc

    def __get__(self, obj, objtype):
        return obj.__dict__.get(self._name, self._default)

    def __set__(self, obj, value):
        if obj is None:
            return self
        for name in self._names:
            obj.__dict__[name] = value

    @property
    def __doc__(self):
        return self._doc


def add_prop(*field_doc_pairs, **defaults):
    def decorator(clz):
        def create(**kwargs):
            instance = clz()
            for key, value in kwargs.items():
                if key not in instance.added_prop:
                    raise ValueError(
                        "Unknown property:%s, valid properties: %s"
                        % (key, instance.added_prop)
                    )
                setattr(instance, key, value)
            return instance
        added_prop = []
        for f in field_doc_pairs:
            names = f[0].split(" ")
            if len(f) > 2:
                default = f[2]
            else:
                default = None
            for name in names:
                if default is None:
                    default = defaults.get(name, None)
                setattr(
                    clz,
                    name,
                    Property(name, names, f[1], default),
                )
            added_prop.extend(names)
        setattr(clz, "added_prop", added_prop)
        setattr(clz, "create", staticmethod(create))
        return clz

    return decorator


@add_prop(
    ("dtype", "data type"),
    ("name", "feature_name"),
    ("is_sparse", "whether it is a sparse or dense feature"),
    ("is_label", "whether it is a label or a feature"),
)
class Column(object):
    @property
    def keys(self):
        return self.added_prop

    def __str__(self):
        result = [k + "=" + str(getattr(self, k)) for k in self.keys]
        return "{" + ";".join(result) + "}"

    def set_default(self):
        pass

    __repr__ = __str__


def parse_features(dataset):
    print("parse_features")
    default_columns_types = []
    default_columns_names = []
    label_column_name = None
    dense_col = [
        Column.create(  # type: ignore
            name="I" + str(i),
            dtype="float32" if i != 0 else "int32",
            is_sparse=False,
            is_label=False if i != 0 else True,
        )
        for i in range(0, 14)
    ]

    sparse_col = [
        Column.create(  # type: ignore
            name="C" + str(i), dtype="string", is_sparse=True, is_label=False
        )
        for i in range(1, 27)
    ]
    columns = dense_col + sparse_col
    for i in columns:
        float_types = ["float", "float32", "float64", "double"]
        int_types = ["int", "int8", "int16", "int32", "int64"]
        uint_types = ["uint8", "uint16", "uint32", "uint64"]
        all_types = float_types + int_types + uint_types
        if i.is_label is True:
            label_column_name = i.name
        dtype = i.dtype
        if dtype == "string":
            default_val = ""
        elif dtype in all_types:
            default_val = np.dtype(dtype).type(0)
        default_columns_types.append(default_val)
        default_columns_names.append(i.name)

    def parse_csv(value):
        columns = parsing_ops.decode_csv(
            value, record_defaults=default_columns_types, field_delim=","
        )
        features = dict(zip(default_columns_names, columns))
        labels = features.pop(label_column_name)
        return features, labels

    return dataset.map(parse_csv, num_parallel_calls=10)


def iterator():
    fd = open("./data_kaggle_ad_ctr_train.csv", "r")
    dataset = fd.readlines()
    print("read")
    for i in range(0, len(dataset)):
        td = dataset[i]
        td = td.strip()
        tdd = td.split(",")
        # assert len(tdd) == 40
        yield td


def make_dataset():
    def reader_fn():
        for item in iterator():
            yield item

    print("building dataset with reader")
    print("Make dataset from reader_fn")

    dataset = tf.data.Dataset.from_generator(
        reader_fn, output_types=tf.string
    )
    dataset = dataset.batch(batch_size).repeat(epoch)
    return parse_features(dataset)


def train_input_fn():
    return make_dataset()


class GlobalStepHook(SessionRunHook):
    def __init__(self, every_n_iter=1):
        self._fetches = dict()
        self._timer = SecondOrStepTimer(every_steps=every_n_iter)
        print("ModelSizeHook: every_n_iter: {}".format(every_n_iter))

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        self._fetches["global_step"] = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        """before_run"""
        session = run_context.session
        global_step = session.run(self._fetches["global_step"])
        print("global_step: {}".format(global_step))

    def end(self, session):
        print("hook end")


if __name__ == '__main__':
    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]
    config = tf.estimator.RunConfig(
        model_dir="./data", save_checkpoints_steps=300, keep_checkpoint_max=10
    )
    classifier = model.DeepFM(model_dir="./data", config=config, params={
        "dense_features": dense_features,
        "sparse_features": sparse_features,
    })
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=100)
    eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    # dense_col = [
    #     Column.create(  # type: ignore
    #         name="I" + str(i),
    #         dtype="float32" if i != 0 else "int32",
    #         is_sparse=False,
    #         is_label=False if i != 0 else True,
    #     )
    #     for i in range(0, 14)
    # ]
    #
    # sparse_col = [
    #     Column.create(  # type: ignore
    #         name="C" + str(i), dtype="string", is_sparse=True, is_label=False
    #     )
    #     for i in range(1, 27)
    # ]
    # col = dense_col + sparse_col
    # print(dense_col + sparse_col)
    # for i in col:
    #     print(i)