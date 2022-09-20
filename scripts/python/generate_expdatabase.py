# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

import sys
import sqlite3
import numpy as np
import os
import argparse
import itertools
import collections

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])

from database import image_ids_to_pair_id, pair_id_to_image_ids, array_to_blob, blob_to_array, COLMAPDatabase

from read_write_model import read_model, CAMERA_MODEL_NAMES

def verify(args):
    if not os.path.exists(args.database_path):
        return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # Read and check cameras.

    cameras = dict(
        (camera_id, [model, width, height, params, prior])
        for camera_id, model, width, height, params, prior in db.execute(
            "SELECT * FROM cameras")
    )
    print(len(cameras))

    # Read and check images.

    images = dict(
        (image_id, name)
        for image_id, name in db.execute(
            "SELECT image_id, name FROM images"))

    print(len(images))

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    print(len(keypoints))

    matches = {}
    for pair_id, data in db.execute("SELECT pair_id, data FROM matches"):
        if data is not None:
            matches[pair_id_to_image_ids(pair_id)] = blob_to_array(data, np.uint32, (-1, 2))

    print(len(matches))

    # Clean up.

    db.close()


def main(args):


    cameras, images, points3D = read_model(path=args.input_model, ext=args.input_format)

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db_to_read = COLMAPDatabase.connect(args.read_database)
    db = COLMAPDatabase.connect(args.database_path, read_only=True)

    # For convenience, try creating all the tables upfront.
    db.create_tables()

    validated_keypoints = collections.defaultdict(list)
    validated_matches = collections.defaultdict(list)
    for point3d_id, point3d_info in sorted(points3D.items()):
        images_ids = point3d_info.image_ids
        keypoints_ids = point3d_info.point2D_idxs
        for image1_id, keypoint1_id in zip(images_ids, keypoints_ids):
            validated_keypoints[image1_id].append(keypoint1_id)
        for (image1_id, image2_id), (keypoint1_id, keypoint2_id) in zip(itertools.combinations(images_ids, 2), itertools.combinations(keypoints_ids, 2)):
            if image1_id > image2_id:
                pair_id = image_ids_to_pair_id(image1_id, image2_id)
                value = [keypoint1_id, keypoint2_id]
            elif image1_id < image2_id:
                pair_id = image_ids_to_pair_id(image2_id, image1_id)
                value = [keypoint2_id, keypoint1_id]
            else:
                continue
            validated_matches[pair_id].append(value)

    for camera_id, model, width, height, params, prior_focal_length in db_to_read.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras"):
        db.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, params, prior_focal_length))

    for image_id, rows, cols, data in db_to_read.execute("SELECT image_id, rows, cols, data FROM descriptors"):
        valid_keypoints_for_image = validated_keypoints[image_id]
        db.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id, len(valid_keypoints_for_image), cols, array_to_blob(blob_to_array(data, np.float32, (-1, 2))[valid_keypoints_for_image, :])))

    for image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz in db_to_read.execute("SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images"):
        db.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz))

    for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in db_to_read.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries"):
        db.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id, rows, cols, data, config, F, E, H, qvec, tvec))

    for image_id, rows, cols, data in db_to_read.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        valid_keypoints_for_image = validated_keypoints[image_id]
        db.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id, len(valid_keypoints_for_image), cols, array_to_blob(blob_to_array(data, np.float32, (-1, 2))[valid_keypoints_for_image, :])))

    for pair_id, rows, cols, data in db_to_read.execute("SELECT pair_id, rows, cols, data FROM matches"):
        if data is not None:
            valid_matches_for_image_pair = validated_matches[pair_id]
            matches_to_keep = np.array([keypoint_pair for keypoint_pair in blob_to_array(data, np.float32, (-1, 2)).tolist() if keypoint_pair in valid_matches_for_image_pair])
            db.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (pair_id, len(matches_to_keep), cols, array_to_blob(matches_to_keep)))
        else:
            db.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (pair_id, rows, cols, data))
    # # for camera_id, camera_info in sorted(cameras.items()):
    # #     id = camera_info.id
    # #     model = CAMERA_MODEL_NAMES[camera_info.model].model_id
    # #     width = camera_info.width
    # #     height = camera_info.height
    # #     params = camera_info.params
    # #     db.add_camera(model, width, height, params, camera_id=id)
    #
    #
    # keypoints_read = dict(
    #     (image_id, blob_to_array(data, np.float32, (-1, 2)))
    #     for image_id, data in db_to_read.execute(
    #         "SELECT image_id, data FROM keypoints"))
    # descriptors_read = dict(
    #     (image_id, blob_to_array(data, np.float32, (-1, 2)))
    #     for image_id, data in db_to_read.execute(
    #         "SELECT image_id, data FROM descriptors"))
    # for image_id, image_info in sorted(images.items()):
    #     id = image_info.id
    #     name = image_info.name
    #     camera = image_info.camera_id
    #     keypoints = image_info.xys
    #     db.add_image(name, camera)
    #     db.add_keypoints(id, keypoints_read[id])
    #     db.add_descriptors(id, descriptors_read[id])
    #
    # matches = collections.defaultdict(list)
    # for point3d_id, point3d_info in sorted(points3D.items()):
    #     images_ids = point3d_info.image_ids
    #     keypoints_ids = point3d_info.point2D_idxs
    #     for (image1_id, image2_id), (keypoint1_id, keypoint2_id) in zip(itertools.combinations(images_ids, 2), itertools.combinations(keypoints_ids, 2)):
    #         if image1_id > image2_id:
    #             pair_id = (image1_id, image2_id)
    #             value = [keypoint1_id, keypoint2_id]
    #         elif image1_id < image2_id:
    #             pair_id = (image2_id, image1_id)
    #             value = [keypoint2_id, keypoint1_id]
    #         else:
    #             continue
    #         matches[pair_id].append(value)
    #
    # for (image1_id, image2_id), keypoints_lists in matches.items():
    #     db.add_matches(int(image1_id), int(image2_id), np.array(keypoints_lists))
    #
    # existing_matches = set(matches.keys())
    # for image1_id, image2_id in itertools.combinations(list(images.keys()), 2):
    #     if image1_id > image2_id:
    #         pair_id = (image1_id, image2_id)
    #     elif image1_id < image2_id:
    #         pair_id = (image2_id, image1_id)
    #     else:
    #         continue
    #     if (pair_id) not in existing_matches:
    #         db.execute(
    #             "INSERT INTO matches VALUES (?, ?, ?, ?)",
    #             (image_ids_to_pair_id(image1_id, image2_id), 0, 0, None,))
    #
    # for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in db_to_read.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries"):
    #     # if elems[1] is not None:
    #     # pair_id, rows, cols, data, config, F, E, H, qvec, tvec = elems
    #     # print(blob_to_array(data, np.uint32, (-1, 2)))
    #     db.execute(
    #         "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    #         (pair_id, rows, cols, data, config, F, E, H, qvec, tvec))
    #     # else:
    #     #     pair_id = elems[0]
    #     #     db.execute(
    #     #         "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    #     #         (pair_id, 0, 0, None, 0, None, None, None, None, None))


    # Commit the data to the file.
    db.commit()

    # Clean up.

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and write COLMAP binary and text models")
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    parser.add_argument("--read_database", help="path to the original database")
    parser.add_argument("--delete", choices=["True", "False"],
                        help="delete database when done", default="False")
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    print("CREATING DATABASE FROM FILES")
    main(args)
    print("GOING TO VERIFY")
    verify(args)
    if args.delete == "True":
        print("Deleting")
        if os.path.exists(args.database_path):
            os.remove(args.database_path)
