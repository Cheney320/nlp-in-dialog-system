#!/usr/bin/env python3
import json
import os
import zipfile
import sys
from collections import OrderedDict




def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, "r")
    return json.load(archive.open(filename))


def preprocess():
    # path
    data_key = ["train", "val", "test"]
    data = {}
    for key in data_key:
        # read crosswoz source data from json.zip
        data[key] = read_zipped_json(
            "../data/raw/" + key + ".json.zip",
            key + ".json",
        )
        print("load {}, size {}".format(key, len(data[key])))

    # generate train, val, tests dataset
    for key in data_key:
        sessions = []
        for no, sess in data[key].items():
            processed_data = OrderedDict()
            processed_data["sys-usr"] = sess["sys-usr"]
            processed_data["type"] = sess["type"]
            processed_data["task description"] = sess["task description"]
            messages = sess["messages"]
            processed_data["turns"] = [
                OrderedDict(
                    {
                        "role": message["role"],
                        "utterance": message["content"],
                        "dialog_act": message["dialog_act"],
                    }
                )
                for message in messages
            ]
            sessions.append(processed_data)
        json.dump(
            sessions,
            open(
                "../data/readable_data/readable_{}_data.json".format(key),
                "w",
                encoding="utf-8",
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )

        print("../data/readable_data/readable_{}_data.json".format(key))


if __name__ == "__main__":
    preprocess()
