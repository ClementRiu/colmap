import sys
import sqlite3
import numpy as np
import os
import argparse
import itertools
import collections

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

# CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
#     camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
#     model INTEGER NOT NULL,
#     width INTEGER NOT NULL,
#     height INTEGER NOT NULL,
#     params BLOB,
#     prior_focal_length INTEGER NOT NULL)"""
#
# CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
#     image_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB,
#     FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
#
# CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
#     image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
#     name TEXT NOT NULL UNIQUE,
#     camera_id INTEGER NOT NULL,
#     prior_qw REAL,
#     prior_qx REAL,
#     prior_qy REAL,
#     prior_qz REAL,
#     prior_tx REAL,
#     prior_ty REAL,
#     prior_tz REAL,
#     CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
#     FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
# """.format(MAX_IMAGE_ID)
#
# CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
# CREATE TABLE IF NOT EXISTS two_view_geometries (
#     pair_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB,
#     config INTEGER NOT NULL,
#     F BLOB,
#     E BLOB,
#     H BLOB,
#     qvec BLOB,
#     tvec BLOB)
# """
#
# CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
#     image_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB,
#     FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
# """
#
# CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
#     pair_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB)"""
#
# CREATE_NAME_INDEX = \
#     "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
#
# CREATE_ALL = "; ".join([
#     CREATE_CAMERAS_TABLE,
#     CREATE_IMAGES_TABLE,
#     CREATE_KEYPOINTS_TABLE,
#     CREATE_DESCRIPTORS_TABLE,
#     CREATE_MATCHES_TABLE,
#     CREATE_TWO_VIEW_GEOMETRIES_TABLE,
#     CREATE_NAME_INDEX
# ])

from database import image_ids_to_pair_id, pair_id_to_image_ids, array_to_blob, blob_to_array, COLMAPDatabase

from read_write_model import read_model, CAMERA_MODEL_NAMES

import new_generate as ng
from collections import defaultdict

class FEATURE:
    def __init__(self,
                 feature_id_, image_id_, point3D_id_, index_,
                 position_, descriptor_,
                 match_right_, match_left_,
                 ncols_keypoints_, ncols_descriptors_):
        self._feature_id = feature_id_
        self._image_id = image_id_
        self._point3D_id = point3D_id_
        self._index = index_

        self._position = position_
        self._descriptor = descriptor_
        self._match_right = match_right_
        self._match_left = match_left_

        self._ncols_keypoints = ncols_keypoints_
        self._ncols_descriptors = ncols_descriptors_

    def is_valid(self):
        return len(self._match_right) + len(self._match_left) > 0

if __name__ == "__main__":
    input_model = "../../../colmap_dataset/gerrard-hall-small-backup/sparse/"
    input_format = ".txt"
    read_database = "../../../colmap_dataset/gerrard-hall-small-backup/database.db"
    database_path = "../../../colmap_dataset/gerrard-hall-small/database.db"
    delete = True

    if delete:
        if os.path.exists(database_path):
            print("Deleting")
            os.remove(database_path)

    cameras_out, images_out, points3D_out = read_model(path=input_model, ext=input_format)

    if os.path.exists(database_path):
        print("ERROR: database path already exists -- will not modify it.")
        assert(False)

    # Open the database.

    db_to_read = COLMAPDatabase.connect(read_database, read_only=True)
    db = COLMAPDatabase.connect(database_path)

    # For convenience, try creating all the tables upfront.
    db.create_tables()

    Cameras = ng.CAMERAS(db_to_read, cameras_out)
    Images = ng.IMAGES(db_to_read, images_out)

    Two_view_geometrys = ng.TWO_VIEW_GEOMETRYS(db_to_read, images_out, True)

    Matches = ng.MATCHES(db_to_read)

    image_right = {}
    image_left = {}
    for id_pair, two_view in Two_view_geometrys._two_view_geometries.items():
        image_id1, image_id2 = two_view._id_pair
        image_right[image_id1] = defaultdict(list)
        image_left[image_id2] = defaultdict(list)
        for id_feature1, id_feature2 in two_view._data:
            image_right[image_id1][id_feature1].append([image_id2, id_feature2])
            image_left[image_id2][id_feature2].append([image_id1, id_feature1])

    Keypoints = ng.KEYPOINTS(db_to_read)
    Descriptors = ng.DESCRIPTORS(db_to_read)

    features_by_image = defaultdict(dict)
    all_features = {}

    id_feature = 0
    for image_id, keypoints in Keypoints._keypoints.items():
        for index in range(keypoints._rows):
            point3D_id = Images._images[image_id]._image_out.point3D_ids[index]
            descriptor = Descriptors._descriptors[image_id]
            match_right = {}
            if image_id in image_right:
                if index in image_right[image_id]:
                    match_right = image_right[image_id][index]
            match_left = {}
            if image_id in image_left:
                if index in image_left[image_id]:
                    match_left = image_left[image_id][index]
            feature = FEATURE(id_feature, image_id, point3D_id, index,
                              keypoints._data[index, :], descriptor._data[index, :],
                              match_right, match_left,
                              keypoints._cols, descriptor._cols
                             )
            features_by_image[image_id][index] = feature
            all_features[id_feature] = feature
            id_feature += 1

    keep = {}
    for image_id, features in features_by_image.items():
        keep[image_id] = set()
        for feature in features.values():
            if feature.is_valid():
                keep[image_id].add(feature._feature_id)


    match_to_right = defaultdict(list)
    match_to_left = defaultdict(list)
    for id_pair, two_view in Two_view_geometrys._two_view_geometries.items():
        image_id1, image_id2 = two_view._id_pair
        for id_feature1, id_feature2 in two_view._data:
            feature1 = features_by_image[image_id1][id_feature1]
            feature2 = features_by_image[image_id2][id_feature2]
            match_to_right[feature1._feature_id].append(feature2._feature_id)
            match_to_left[feature2._feature_id].append(feature1._feature_id)

    for feature_id, feature in all_features.items():
        feature._match_right = match_to_right[feature_id]
        feature._match_left = match_to_left[feature_id]

    descriptors_feated = defaultdict(list)
    keypoints_feated = defaultdict(list)
    matches_feated = defaultdict(list)
    two_view_geometry_feated = defaultdict(list)
    new_feature_by_image = defaultdict(dict)
    for image_id, features_to_keep in keep.items():
        for index_feat, feature_id in enumerate(features_to_keep):
            feature = all_features[feature_id]
            feature._index = index_feat
            new_feature_by_image[image_id][index_feat] = feature
            descriptors_feated[image_id].append([feature._descriptor, feature._ncols_descriptors])
            keypoints_feated[image_id].append([feature._position, feature._ncols_keypoints])

    for image_id, features_to_keep in keep.items():
        for index_feat, feature_id in enumerate(features_to_keep):
            feature = all_features[feature_id]
            for feature_id2 in feature._match_right:
                feature2 = all_features[feature_id2]
                image_id1 = feature._image_id
                image_id2 = feature2._image_id
                index1 = feature._index
                index2 = feature2._index
                if image_id1 > image_id2:
                    value = [index2, index1]
                else:
                    value = [index1, index2]
                pair_id = image_ids_to_pair_id(image_id1, image_id2)
                two_view_geometry_feated[pair_id].append(value)
                matches_feated[pair_id].append(value)
            for feature_id0 in feature._match_left:
                feature0 = all_features[feature_id0]
                image_id1 = feature._image_id
                image_id0 = feature0._image_id
                index1 = feature._index
                index0 = feature0._index
                if image_id0 > image_id1:
                    value = [index1, index0]
                else:
                    value = [index0, index1]
                pair_id = image_ids_to_pair_id(image_id0, image_id1)
                two_view_geometry_feated[pair_id].append(value)
                matches_feated[pair_id].append(value)

    descriptors_feated_curated = {}
    for image_id, descriptors in descriptors_feated.items():
        num_row = len(descriptors)
        num_col = descriptors[0][1]
        descript_as_array = np.zeros((num_row, num_col))
        for index, descriptor in enumerate(descriptors):
            if descriptor[1] != num_col:
                print("ERROR")
            else:
                descript_as_array[index, :] = descriptor[0]
        descriptors_feated_curated[image_id] = ng.DESCRIPTOR(image_id,
                                                             num_row, num_col,
                                                             array_to_blob(descript_as_array.astype(np.int8)))
    keypoints_feated_curated = {}
    for image_id, keypoints in keypoints_feated.items():
        num_row = len(keypoints)
        num_col = keypoints[0][1]
        keypoint_as_array = np.zeros((num_row, num_col))
        for index, keypoint in enumerate(keypoints):
            if keypoint[1] != num_col:
                print("ERROR")
            else:
                keypoint_as_array[index, :] = keypoint[0]
        keypoints_feated_curated[image_id] = ng.KEYPOINT(image_id,
                                                         num_row, num_col,
                                                         array_to_blob(keypoint_as_array.astype(np.float32)))

    matches_feated_curated = {}
    for pair_id, matches in matches_feated.items():
        num_row = len(matches)
        num_col = 2
        match_as_array = np.array(matches)
        matches_feated_curated[pair_id] = ng.MATCH(pair_id,
                                                   num_row, num_col,
                                                   array_to_blob(match_as_array.astype(np.int32)))

    twoview_feated_curated = {}
    for pair_id, two_view in two_view_geometry_feated.items():
        num_row = len(two_view)
        num_col = 2
        match_as_array = np.array(two_view)
        two_view_init = Two_view_geometrys._two_view_geometries[pair_id]
        twoview_feated_curated[pair_id] = ng.TWO_VIEW_GEOMETRY(pair_id,
                                                   num_row, num_col,
                                                   match_as_array,
                                                   two_view_init._config,
                                                   two_view_init._F, two_view_init._E, two_view_init._H,
                                                   two_view_init._qvec, two_view_init._tvec)

    for camera_id in Cameras._cameras:
        Cameras._cameras[camera_id].write(db)

    for image_id in Images._images:
        Images._images[image_id].write(db)

    for image_id in descriptors_feated_curated:
        descriptors_feated_curated[image_id].write(db)

    for image_id in keypoints_feated_curated:
        keypoints_feated_curated[image_id].write(db)

    for pair_id in twoview_feated_curated:
        twoview_feated_curated[pair_id].write(db)

    for pair_id in matches_feated_curated:
        matches_feated_curated[pair_id].write(db)
