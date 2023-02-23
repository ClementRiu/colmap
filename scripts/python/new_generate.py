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

from scipy.spatial.transform import Rotation as Rot

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

from generate_expdatabase import verify

def get_calib(camera_):
    camera_type = camera_.model
    params = camera_.params
    calibration = np.eye(3)
    if camera_type in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        calibration[0, 0] = params[0]
        calibration[1, 1] = params[0]
        calibration[0, 2] = params[1]
        calibration[1, 2] = params[2]
    elif camera_type in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        calibration[0, 0] = params[0]
        calibration[1, 1] = params[1]
        calibration[0, 2] = params[2]
        calibration[1, 2] = params[3]
    else:
        print("Error, camera not recognised")
    return calibration

class CAMERA:
    def __init__(self, camera_id_, model_, width_, height_, params_, prior_focal_length_, camera_out_):
        self._camera_id = camera_id_
        self._model = model_
        self._width = width_
        self._height = height_
        self._params = params_
        self._prior_focal_length = prior_focal_length_

        self._camera_out = camera_out_
        self._calib = get_calib(camera_out_)

    def write(self, db_to_write_):
        db_to_write_.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (self._camera_id, self._model, self._width, self._height, self._params, self._prior_focal_length))

class DESCRIPTOR:
    def __init__(self, image_id_, rows_, cols_, data_):
        self._image_id = image_id_
        self._rows = rows_
        self._cols = cols_
        self._data = np.frombuffer(data_, dtype=np.int8).reshape(rows_, cols_)

    def write(self, db_to_write_):
        db_to_write_.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (self._image_id, self._rows, self._cols, array_to_blob(self._data)))

class IMAGE:
    def __init__(self, image_id_, name_, camera_id_,
                prior_qw_, prior_qx_, prior_qy_, prior_qz_,
                prior_tx_, prior_ty_, prior_tz_, image_out_):
        self._image_id = image_id_
        self._name = name_
        self._camera_id = camera_id_
        self._prior_qw = prior_qw_
        self._prior_qx = prior_qx_
        self._prior_qy = prior_qy_
        self._prior_qz = prior_qz_
        self._prior_tx = prior_tx_
        self._prior_ty = prior_ty_
        self._prior_tz = prior_tz_

        self._image_out = image_out_
        self._rotation = Rot.from_quat(image_out_.qvec[[1, 2, 3, 0]]).as_matrix()
        self._translation = image_out_.tvec

    def write(self, db_to_write_):
        db_to_write_.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._image_id, self._name, self._camera_id, self._prior_qw, self._prior_qx, self._prior_qy, self._prior_qz, self._prior_tx, self._prior_ty, self._prior_tz))

class TWO_VIEW_GEOMETRY:
    def __init__(self, pair_id_, rows_, cols_, data_processed_, config_, F_, E_, H_, qvec_, tvec_):
        self._pair_id = pair_id_
        self._rows = rows_
        self._cols = cols_
        self._data = data_processed_
        self._config = config_
        self._F = np.frombuffer(F_, dtype=np.float64).reshape(3, 3)
        self._E = np.frombuffer(E_, dtype=np.float64).reshape(3, 3)
        self._H = np.frombuffer(H_, dtype=np.float64).reshape(3, 3)
        self._qvec = np.frombuffer(qvec_, dtype=np.float64)
        self._tvec = np.frombuffer(tvec_, dtype=np.float64)

        self._id_pair = pair_id_to_image_ids(pair_id_)

    def write(self, db_to_write_):
        db_to_write_.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._pair_id, self._rows, self._cols, array_to_blob(self._data), self._config, array_to_blob(self._F), array_to_blob(self._E), array_to_blob(self._H), array_to_blob(self._qvec), array_to_blob(self._tvec)))

class KEYPOINT:
    def __init__(self, image_id_, rows_, cols_, data_):
        self._image_id = image_id_
        self._rows = rows_
        self._cols = cols_
        self._data = np.frombuffer(bytearray(data_), dtype=np.float32).reshape(rows_, cols_)

        self._corrected = set()

    def write(self, db_to_write_):
        db_to_write_.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (self._image_id, self._rows, self._cols, array_to_blob(self._data)))

class MATCH:
    def __init__(self, pair_id_, rows_, cols_, data_):
        self._pair_id = pair_id_
        self._rows = rows_
        self._cols = cols_
        self._data = np.frombuffer(data_, dtype=np.int32).reshape(rows_, cols_)

        self._id_pair = pair_id_to_image_ids(pair_id_)

    def write(self, db_to_write_):
        db_to_write_.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (self._pair_id, self._rows, self._cols, array_to_blob(self._data)))

class CAMERAS:
    def __init__(self, db_to_read_, cameras_out_):
        self._cameras = {}
        for camera_id, model, width, height, params, prior_focal_length in db_to_read_.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras"):
            camera_out = cameras_out_[camera_id]
            self._cameras[camera_id] = CAMERA(camera_id, model, width, height, params, prior_focal_length, camera_out)

    def write_to_base(self, db_to_write_):
        for camera_id in self._cameras:
            self._cameras[camera_id].write(db_to_write_)

class DESCRIPTORS:
    def __init__(self, db_to_read_):
        self._descriptors = {}
        for image_id, rows, cols, data in db_to_read_.execute("SELECT image_id, rows, cols, data FROM descriptors"):
            self._descriptors[image_id] = DESCRIPTOR(image_id, rows, cols, data)

    def write_to_base(self, db_to_write_):
        for image_id in self._descriptors:
            self._descriptors[image_id].write(db_to_write_)

class IMAGES:
    def __init__(self, db_to_read_, images_out_):
        self._images = {}
        for image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz in db_to_read_.execute("SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images"):
            image_out = images_out_[image_id]
            assert(image_out.name == name)
            self._images[image_id] = IMAGE(image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz, image_out)

    def write_to_base(self, db_to_write_):
        for image_id in self._images:
            self._images[image_id].write(db_to_write_)

class TWO_VIEW_GEOMETRYS:
    def __init__(self, db_to_read_, images_, validate_=False):
        self._two_view_geometries = {}
        for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in db_to_read_.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries"):
            data = np.frombuffer(data, dtype=np.int32).reshape(rows, cols)
            if validate_:
                id_pair = pair_id_to_image_ids(pair_id)
                image1 = images_[id_pair[0]]
                image2 = images_[id_pair[1]]
                valid_data = []
                for feature_idx1, feature_idx2 in data:
                    if image1.point3D_ids[feature_idx1] !=-1 and image2.point3D_ids[feature_idx2] != -1:
                        valid_data.append([feature_idx1, feature_idx2])
                valid_data = np.array(valid_data, dtype=np.int32)
                valid_rows = len(valid_data)
            else:
                valid_data = data
                valid_rows = rows
            self._two_view_geometries[pair_id] = TWO_VIEW_GEOMETRY(pair_id, valid_rows, cols, valid_data, config, F, E, H, qvec, tvec)

    def write_to_base(self, db_to_write_):
        for pair_id in self._two_view_geometries:
            self._two_view_geometries[pair_id].write(db_to_write_)

class KEYPOINTS:
    def __init__(self, db_to_read_):
        self._keypoints = {}
        for image_id, rows, cols, data in db_to_read_.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            self._keypoints[image_id] = KEYPOINT(image_id, rows, cols, data)

    def _align(self, xyz_val_, camera_, image_, safe_ = True):
        uv_val = camera_._calib.dot(image_._rotation.dot(xyz_val_) + image_._translation)
        if uv_val[2] > 0 or (uv_val[2] < 0 and not safe_):
            uv_val = uv_val / uv_val[2]
        else:
            print("ERROR: point not in front of camera")

        return uv_val[:2]

    def _add_noise(self, uv_val_, noise_std_, camera_):
        noisy_uv_val = uv_val_ + np.random.uniform(-noise_std_, noise_std_, 2)
        truncated = np.clip(noisy_uv_val, a_min = [0, 0], a_max = [camera_._width, camera_._height])
        return truncated

    def generate_GT_inliers(self, two_view_geometries_, images_, cameras_, points3D_, noise_std_ = 0):
        for id_pair, two_view_geometrie in two_view_geometries_._two_view_geometries.items():
            for id_side, id_image in enumerate(two_view_geometrie._id_pair):
                image = images_._images[id_image]
                camera = cameras_._cameras[image._camera_id]
                keypoint = self._keypoints[id_image]
                idx_features = two_view_geometrie._data[:, id_side]
                for idx_feature in idx_features:
                    if idx_feature in keypoint._corrected:
                        continue
                    xyz_val = points3D_[image._image_out.point3D_ids[idx_feature]].xyz
                    uv_val = self._align(xyz_val, camera, image)
                    if noise_std_ > 0:
                        uv_val = self._add_noise(uv_val, noise_std_, camera)
                    keypoint._data[idx_feature, :2] = uv_val
                    keypoint._corrected.add(idx_feature)

    def generate_GT_outliers(self, points3D_, cameras_, images_):
        return

    def write_to_base(self, db_to_write_):
        for image_id in self._keypoints:
            self._keypoints[image_id].write(db_to_write_)

class MATCHES:
    def __init__(self, db_to_read_):
        self._matches = []
        for pair_id, rows, cols, data in db_to_read_.execute("SELECT pair_id, rows, cols, data FROM matches"):
            self._matches.append(MATCH(pair_id, rows, cols, data))

    def write_to_base(self, db_to_write_):
        for match in self._matches:
            match.write(db_to_write_)

def main(args):
    cameras_out, images_out, points3D_out = read_model(path=args.input_model, ext=args.input_format)

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db_to_read = COLMAPDatabase.connect(args.read_database, read_only=True)
    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.
    db.create_tables()

    Cameras = CAMERAS(db_to_read, cameras_out)
    Descriptors = DESCRIPTORS(db_to_read)
    Images = IMAGES(db_to_read, images_out)
    Two_view_geometrys = TWO_VIEW_GEOMETRYS(db_to_read, images_out, args.validate == "True")
    Keypoints = KEYPOINTS(db_to_read)
    Matches = MATCHES(db_to_read)

    if args.align =="True":
        Keypoints.generate_GT_inliers(Two_view_geometrys, Images, Cameras, points3D_out, args.noise_std)

    Cameras.write_to_base(db)
    Descriptors.write_to_base(db)
    Images.write_to_base(db)
    Two_view_geometrys.write_to_base(db)
    Keypoints.write_to_base(db)
    Matches.write_to_base(db)

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

    parser.add_argument("--validate", choices=["True", "False"],
                        help="Validate points with 3D", default="False")
    parser.add_argument("--align", choices=["True", "False"],
                        help="Align points with 3D", default="False")
    parser.add_argument("--noise_std", type=float,
                        help="Noise level of inliers", default=0.0)
    args = parser.parse_args()

    print("CREATING DATABASE FROM FILES")
    main(args)
    print("GOING TO VERIFY")
    verify(args)
    if args.delete == "True":
        print("Deleting")
        if os.path.exists(args.database_path):
            os.remove(args.database_path)
