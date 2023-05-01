import sys
import sqlite3
import numpy as np
import os
import argparse
import itertools
import collections

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

from database import image_ids_to_pair_id, pair_id_to_image_ids, array_to_blob, blob_to_array, COLMAPDatabase

from read_write_model import read_model, CAMERA_MODEL_NAMES

import new_generate as ng
from collections import defaultdict

from generate_expdatabase import verify


class FEATURE:
    def __init__(self,
                 feature_id_, image_id_, point3D_id_, index_,
                 position_, descriptor_,
                 match_right_, match_left_,
                 ncols_keypoints_, ncols_descriptors_,
                 inlier_=1):
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

        self._inlier = inlier_

    def is_valid(self):
        return len(self._match_right) + len(self._match_left) > 0

    def print(self):
        print(self._feature_id, self._image_id, self._point3D_id, self._index)
        print('')
        print(self._position)
        print('')
        print(self._match_right, self._match_left)

def assign_match_by_id(Two_view_geometrys_):
    image_right = defaultdict(lambda: defaultdict(list))
    image_left = defaultdict(lambda: defaultdict(list))
    for id_pair, two_view in Two_view_geometrys_._two_view_geometries.items():
        image_id1, image_id2 = two_view._id_pair
        for id_feature1, id_feature2 in two_view._data:
            image_right[image_id1][id_feature1].append([image_id2, id_feature2])
            image_left[image_id2][id_feature2].append([image_id1, id_feature1])

    return image_right, image_left

def create_individual_feature(Keypoints_, Images_, Descriptors_, image_right_, image_left_):
    features_by_image = defaultdict(dict)
    all_features = {}

    id_feature = 0
    for image_id, keypoints in Keypoints_._keypoints.items():
        if Images_._images[image_id]._image_out is None:
            continue
        for index in range(keypoints._rows):
            point3D_id = Images_._images[image_id]._image_out.point3D_ids[index]
            descriptor = Descriptors_._descriptors[image_id]
            match_right = {}
            if image_id in image_right_:
                if index in image_right_[image_id]:
                    match_right = image_right_[image_id][index]
            match_left = {}
            if image_id in image_left_:
                if index in image_left_[image_id]:
                    match_left = image_left_[image_id][index]
            feature = FEATURE(id_feature, image_id, point3D_id, index,
                              keypoints._data[index, :], descriptor._data[index, :],
                              match_right, match_left,
                              keypoints._cols, descriptor._cols
                             )
            features_by_image[image_id][index] = feature
            all_features[id_feature] = feature
            id_feature += 1

    return features_by_image, all_features

def get_valid_features(features_by_image_):
    keep = {}
    for image_id, features in features_by_image_.items():
        keep[image_id] = set()
        for feature in features.values():
            if feature.is_valid():
                keep[image_id].add(feature._feature_id)
    return keep

def correct_match_ids(Two_view_geometrys_, features_by_image_, all_features_):
    match_to_right = defaultdict(list)
    match_to_left = defaultdict(list)
    for id_pair, two_view in Two_view_geometrys_._two_view_geometries.items():
        image_id1, image_id2 = two_view._id_pair
        for id_feature1, id_feature2 in two_view._data:
            feature1 = features_by_image_[image_id1][id_feature1]
            feature2 = features_by_image_[image_id2][id_feature2]
            match_to_right[feature1._feature_id].append(feature2._feature_id)
            match_to_left[feature2._feature_id].append(feature1._feature_id)

    for feature_id, feature in all_features_.items():
        feature._match_right = match_to_right[feature_id]
        feature._match_left = match_to_left[feature_id]

    return

def create_descriptors_and_keypoints_tabs_from_features(keep_, all_features_):
    descriptors_feated = defaultdict(list)
    keypoints_feated = defaultdict(list)
    inlier_outlier = defaultdict(list)
    reordered_feature_by_image = defaultdict(dict)
    for image_id, features_to_keep in keep_.items():
        for index_feat, feature_id in enumerate(features_to_keep):
            feature = all_features_[feature_id]
            feature._index = index_feat
            reordered_feature_by_image[image_id][index_feat] = feature
            descriptors_feated[image_id].append([feature._descriptor, feature._ncols_descriptors])
            keypoints_feated[image_id].append([feature._position, feature._ncols_keypoints])
            inlier_outlier[image_id].append([image_id, feature._index, feature._inlier])

    return descriptors_feated, keypoints_feated, reordered_feature_by_image, inlier_outlier

def create_match_and_2view_tabs_from_features(keep_, all_features_):
    matches_feated = defaultdict(list)
    two_view_geometry_feated = defaultdict(list)

    for image_id, features_to_keep in keep_.items():
        for index_feat, feature_id in enumerate(features_to_keep):
            feature = all_features_[feature_id]
            for feature_id2 in feature._match_right:
                feature2 = all_features_[feature_id2]
                image_id1 = feature._image_id
                image_id2 = feature2._image_id
                index1 = feature._index
                index2 = feature2._index
                if image_id1 > image_id2:
                    value = [index2, index1]
                else:
                    value = [index1, index2]
                pair_id = image_ids_to_pair_id(image_id1, image_id2)
                if value not in two_view_geometry_feated[pair_id]:
                    two_view_geometry_feated[pair_id].append(value)
                if value not in matches_feated[pair_id]:
                    matches_feated[pair_id].append(value)
            for feature_id0 in feature._match_left:
                feature0 = all_features_[feature_id0]
                image_id1 = feature._image_id
                image_id0 = feature0._image_id
                index1 = feature._index
                index0 = feature0._index
                if image_id0 > image_id1:
                    value = [index1, index0]
                else:
                    value = [index0, index1]
                pair_id = image_ids_to_pair_id(image_id0, image_id1)
                if value not in two_view_geometry_feated[pair_id]:
                    two_view_geometry_feated[pair_id].append(value)
                if value not in matches_feated[pair_id]:
                    matches_feated[pair_id].append(value)

    return matches_feated, two_view_geometry_feated

def create_new_descriptor_arrays(descriptors_feated_):
    descriptors_feated_asarray = {}
    for image_id, descriptors in descriptors_feated_.items():
        num_row = len(descriptors)
        num_col = descriptors[0][1]
        descript_as_array = np.zeros((num_row, num_col))
        for index, descriptor in enumerate(descriptors):
            if descriptor[1] != num_col:
                print("ERROR")
            else:
                descript_as_array[index, :] = descriptor[0]
        descriptors_feated_asarray[image_id] = ng.DESCRIPTOR(image_id,
                                                             num_row, num_col,
                                                             descript_as_array.astype(np.int8))
    return descriptors_feated_asarray

def create_new_keypoint_asarray(keypoints_feated_):
    keypoints_feated_asarray = {}
    for image_id, keypoints in keypoints_feated_.items():
        num_row = len(keypoints)
        num_col = keypoints[0][1]
        keypoint_as_array = np.zeros((num_row, num_col))
        for index, keypoint in enumerate(keypoints):
            if keypoint[1] != num_col:
                print("ERROR")
            else:
                keypoint_as_array[index, :] = keypoint[0]
        keypoints_feated_asarray[image_id] = ng.KEYPOINT(image_id,
                                                         num_row, num_col,
                                                         keypoint_as_array.astype(np.float32))
    return keypoints_feated_asarray

def create_new_matches_asarray(matches_feated_, Matches_in_):
    matches_feated_asarray = {}
    for match_in in Matches_in_._matches:
        pair_id = match_in._pair_id
        matches = matches_feated_[pair_id]
        num_row = len(matches)
        num_col = 2
        match_as_array = np.array(matches, dtype=np.int32).reshape(num_row, num_col)
        matches_feated_asarray[pair_id] = ng.MATCH(pair_id,
                                                   num_row, num_col,
                                                   match_as_array)

    return matches_feated_asarray

def create_new_2view_asarray(two_view_geometry_feated_, Two_view_geometrys_):
    twoview_feated_asarray = {}
    for pair_id in Two_view_geometrys_._two_view_geometries.keys():
        two_view = two_view_geometry_feated_[pair_id]
        num_row = len(two_view)
        num_col = 2
        twoview_as_array = np.array(two_view, dtype=np.int32).reshape(num_row, num_col)
        two_view_init = Two_view_geometrys_._two_view_geometries[pair_id]
        twoview_feated_asarray[pair_id] = ng.TWO_VIEW_GEOMETRY(pair_id,
                                                   num_row, num_col,
                                                   twoview_as_array,
                                                   two_view_init._config,
                                                   two_view_init._F, two_view_init._E, two_view_init._H,
                                                   two_view_init._qvec, two_view_init._tvec)
    return twoview_feated_asarray

def write_descriptors_to_base(descriptors_feated_asarray_, db_):
    for image_id in descriptors_feated_asarray_:
        descriptors_feated_asarray_[image_id].write(db_)
    return

def write_keypoints_to_base(keypoints_feated_asarray_, db_):
    for image_id in keypoints_feated_asarray_:
        keypoints_feated_asarray_[image_id].write(db_)
    return

def write_twoview_to_base(twoview_feated_asarray_, db_):
    for pair_id in twoview_feated_asarray_:
        twoview_feated_asarray_[pair_id].write(db_)
    return

def write_matches_to_base(matches_feated_asarray_, db_):
    for pair_id in matches_feated_asarray_:
        matches_feated_asarray_[pair_id].write(db_)
    return

def write_inlier_outlier(inlier_outlier_, path_):
    data = []
    for image_id in inlier_outlier_:
        for elem in inlier_outlier_[image_id]:
            data.append(elem)
    np.savetxt(path_, np.array(data))
    return

def _align(xyz_val_, camera_, image_, safe_ = True):
    uv_val = camera_._calib.dot(image_._rotation.dot(xyz_val_) + image_._translation)
    if uv_val[2] > 0 or (uv_val[2] < 0 and not safe_):
        uv_val = uv_val / uv_val[2]
    else:
        print("ERROR: point not in front of camera")

    return uv_val[:2]

def _add_noise(uv_val_, noise_std_, camera_):
    noisy_uv_val = uv_val_ + np.random.uniform(-noise_std_, noise_std_, 2)
    truncated = np.clip(noisy_uv_val, a_min = [0, 0], a_max = [camera_._width, camera_._height])
    return truncated

def generate_GT_inliers(keep_, all_features_, Images_, Cameras_, points3D_out_, noise_std_, init_image1 = -1, init_image2 = -1):
    max_pert = 0
    for image_id, features_to_keep in keep_.items():
        for feature_id in features_to_keep:
            feature = all_features_[feature_id]
            image = Images_._images[feature._image_id]
            camera = Cameras_._cameras[image._camera_id]
            xyz_val = points3D_out_[feature._point3D_id].xyz
            uv_val = _align(xyz_val, camera, image)
            if image_id == init_image1 or image_id == init_image2:
                continue
            if noise_std_ > 0:
                uv_val_pert = _add_noise(uv_val, noise_std_, camera)
                pert = np.linalg.norm(uv_val - uv_val_pert)
                uv_val = uv_val_pert
                if pert > max_pert:
                    max_pert = pert
            feature._position = np.hstack([uv_val, feature._position[2:]])
    return max_pert

def create_2view_featured(Two_view_geometrys_, features_by_image_):
    two_views_withfeatures = {}
    for id_pair, two_view in Two_view_geometrys_._two_view_geometries.items():
        image_id1, image_id2 = two_view._id_pair
        two_views_matchs = []

        for id_feature1, id_feature2 in two_view._data:
            feature1 = features_by_image_[image_id1][id_feature1]
            feature2 = features_by_image_[image_id2][id_feature2]
            two_views_matchs.append([feature1._feature_id, feature2._feature_id])
        two_views_withfeatures[id_pair] = two_views_matchs

    return two_views_withfeatures

def _generate_outlier(xyz_val_, camera_, image_, min_offset_, iteration_, max_iter_outlier_):
    width = camera_._width
    height = camera_._height

    uv_val = _align(xyz_val_, camera_, image_, False)
    u, v = uv_val

    if u > 0 and v > 0 and u < width and v < height:
        theta1 = np.arctan((height - v) / (width - u))
        theta2 = np.arctan(u / (height - v)) + np.pi / 2
        theta3 = np.arctan(v / u) + np.pi
        theta4 = np.arctan((width - u) / v) + np.pi * 3 / 2

        assert(theta1 > 0 and theta1 < np.pi / 2)
        assert(theta2 > np.pi / 2 and theta2 < np.pi)
        assert(theta3 > np.pi and theta3 < np.pi * 3 / 2)
        assert(theta4 > np.pi * 3 / 2 and theta4 < 2 * np.pi)
        while iteration_ < max_iter_outlier_:

            theta = np.random.uniform(0, 2 * np.pi)

            if theta >= theta4 or theta < theta1:
                max_offset = (width - u) / np.cos(theta)
            elif theta >= theta1 and theta < theta2:
                max_offset = (height - v) / np.cos(theta - np.pi / 2)
            elif theta >= theta2 and theta < theta3:
                max_offset = u / np.cos(theta - np.pi)
            elif theta >= theta3 and theta < theta4:
                max_offset = v / np.cos(theta - np.pi * 3 / 2)

            if max_offset >= min_offset_:
                offset = np.random.uniform(min_offset_, max_offset)

                u += np.cos(theta) * offset
                v += np.sin(theta) * offset

                assert(u >= 0 and u <= width)
                assert(v >= 0 and v <= height)

                iteration_ += 1
                return np.array([u, v])
            iteration_ += 1
    iteration_ += 1
    return None

def generate_GT_outliers(all_features_, keep_, two_views_withfeatures_,
                         outlier_ratio_, max_inlier_error_, max_try_outlier_,
                         Images_, Cameras_, points3D_out_,
                         init_image1 = -1, init_image2 = -1):
    id_feature_new = len(all_features_)
    for id_pair, two_view in two_views_withfeatures_.items():
        image_ids = pair_id_to_image_ids(id_pair)
        if init_image1 in image_ids and init_image2 in image_ids:
            continue
        elif init_image1 == image_ids[0] or init_image2 == image_ids[0]:
            side = 1
        elif init_image1 == image_ids[1] or init_image2 == image_ids[1]:
            side = 0
        else:
            side = np.random.randint(0, 2)

        num_matches = len(two_view)
        if outlier_ratio_ < 1:
            num_outlier_req = np.floor(outlier_ratio_ * num_matches).astype(int)
        else:
            num_outlier_req = np.floor(outlier_ratio_).astype(int)

        outliers_idx = np.random.choice(num_matches, num_outlier_req, replace=False)

        other_side = 1 - side
        image_id = image_ids[side]
        image = Images_._images[image_id]
        camera = Cameras_._cameras[image._camera_id]

        iteration = 0
        for outlier_id in outliers_idx:
            match_to_modify = two_view[outlier_id]
            feature_to_modify = all_features_[match_to_modify[side]]
            feature_unchanged = all_features_[match_to_modify[other_side]]

            xyz_val = points3D_out_[feature_to_modify._point3D_id].xyz
            outlier_pos = _generate_outlier(xyz_val, camera, image,
                                            max_inlier_error_, iteration, max_try_outlier_ + num_outlier_req)

            if outlier_pos is not None:
                position = np.hstack([outlier_pos, feature_to_modify._position[2:]])

                if side:
                    idx_to_change_side = \
                        np.where(np.array(feature_to_modify._match_left) == feature_unchanged._feature_id)[0][0]
                    feature_to_modify._match_left.pop(idx_to_change_side)

                    idx_to_change_otherside =  \
                        np.where(np.array(feature_unchanged._match_right) == feature_to_modify._feature_id)[0][0]
                    feature_unchanged._match_right[idx_to_change_otherside] = id_feature_new

                    match_right = []
                    match_left = [feature_unchanged._feature_id]
                else:
                    idx_to_change_side =  \
                        np.where(np.array(feature_to_modify._match_right) == feature_unchanged._feature_id)[0][0]

                    feature_to_modify._match_right.pop(idx_to_change_side)

                    idx_to_change_otherside =  \
                        np.where(np.array(feature_unchanged._match_left) == feature_to_modify._feature_id)[0][0]
                    feature_unchanged._match_left[idx_to_change_otherside] = id_feature_new

                    match_right = [feature_unchanged._feature_id]
                    match_left = []

                new_feature = FEATURE(id_feature_new, image_id, feature_to_modify._point3D_id, -1,
                              position, feature_to_modify._descriptor,
                              match_right, match_left,
                              feature_to_modify._ncols_keypoints, feature_to_modify._ncols_descriptors,
                              0
                             )

                all_features_[id_feature_new] = new_feature
                keep_[image_id].add(id_feature_new)
                id_feature_new += 1
    return None

def check_duplicate(keypoints_feated_asarray_,
                    matches_feated_asarray_, twoview_feated_asarray_,
                    verbose_=False):
    max_diff = 0
    if verbose_:
        print("Keypoints:")
    for index, elem in keypoints_feated_asarray_.items():
        diff = len(elem._data) - len(np.unique(elem._data, axis=0))
        if max_diff < diff:
            max_diff = diff
        if verbose_:
            print(index, diff)
    if verbose_:
        print("Matches:")
    for index, elem in matches_feated_asarray_.items():
        diff = len(elem._data) - len(np.unique(elem._data, axis=0))
        if max_diff < diff:
            max_diff = diff
        if verbose_:
            print(pair_id_to_image_ids(index), diff)
    if verbose_:
        print("Two View:")
    for index, elem in twoview_feated_asarray_.items():
        diff = len(elem._data) - len(np.unique(elem._data, axis=0))
        if max_diff < diff:
            max_diff = diff
        if verbose_:
            print(pair_id_to_image_ids(index), diff)
    return max_diff

def main(args):
    ### Import output of some run.
    cameras_out, images_out, points3D_out = read_model(path=args.input_model, ext=args.input_format)

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db_to_read = COLMAPDatabase.connect(args.read_database, read_only=True)
    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.
    db.create_tables()

    ### Load data
    Cameras_in = ng.CAMERAS(db_to_read, cameras_out)
    Images_in = ng.IMAGES(db_to_read, images_out)

    Two_view_geometrys_in = ng.TWO_VIEW_GEOMETRYS(db_to_read, images_out, args.validate == "True")

    Matches_in = ng.MATCHES(db_to_read) # Technically useless.

    image_right, image_left = assign_match_by_id(Two_view_geometrys_in)

    Keypoints_in = ng.KEYPOINTS(db_to_read)
    Descriptors_in = ng.DESCRIPTORS(db_to_read)

    features_by_image = defaultdict(dict)

    features_by_image, all_features = create_individual_feature(Keypoints_in,
                                                                Images_in,
                                                                Descriptors_in,
                                                                image_right,
                                                                image_left)

    keep = get_valid_features(features_by_image)


    correct_match_ids(Two_view_geometrys_in, features_by_image, all_features)

    ### Data is loaded and ready for processing.
    max_inlier_error = 0
    if args.align =="True":
        max_inlier_error = generate_GT_inliers(keep, all_features, Images_in, Cameras_in, points3D_out, args.noise_std, args.init_image1, args.init_image2)

    print("=============  Before =============")
    before = {}
    for image_id_print, feature in keep.items():
        print(image_id_print, len(feature))
        before[image_id_print] = len(feature)

    if args.outlier_ratio > 0:
        two_views_withfeatures = create_2view_featured(Two_view_geometrys_in, features_by_image)

        generate_GT_outliers(all_features, keep, two_views_withfeatures,
                             args.outlier_ratio, max_inlier_error, args.max_try_outlier,
                             Images_in, Cameras_in, points3D_out,
                             args.init_image1, args.init_image2)

    print("============= After =============")
    for image_id_print, feature in keep.items():
        print(image_id_print, len(feature), (len(feature) - before[image_id_print]) / len(feature))

    ### Prepare for export of data.
    descriptors_feated, keypoints_feated, reordered_feature_by_image, inlier_outlier = create_descriptors_and_keypoints_tabs_from_features(keep, all_features)

    matches_feated, two_view_geometry_feated = create_match_and_2view_tabs_from_features(keep, all_features)

    descriptors_feated_asarray = create_new_descriptor_arrays(descriptors_feated)
    keypoints_feated_asarray = create_new_keypoint_asarray(keypoints_feated)
    matches_feated_asarray = create_new_matches_asarray(matches_feated, Matches_in)
    twoview_feated_asarray = create_new_2view_asarray(two_view_geometry_feated, Two_view_geometrys_in)

    duplicate_max = check_duplicate(keypoints_feated_asarray, matches_feated_asarray, twoview_feated_asarray)
    if duplicate_max == 0:

        Cameras_in.write_to_base(db)
        Images_in.write_to_base(db)
        write_descriptors_to_base(descriptors_feated_asarray, db)
        write_keypoints_to_base(keypoints_feated_asarray, db)
        write_matches_to_base(matches_feated_asarray, db)
        write_twoview_to_base(twoview_feated_asarray, db)

        write_inlier_outlier(inlier_outlier, args.inlier_outlier_path)

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
    parser.add_argument("--database_path", default="database.db")
    parser.add_argument("--inlier_outlier_path", help="path to output inlier / outlier marks")


    parser.add_argument("--delete", choices=["True", "False"],
                        help="delete database when done", default="False")

    parser.add_argument("--validate", choices=["True", "False"],
                        help="Validate points with 3D", default="False")
    parser.add_argument("--align", choices=["True", "False"],
                        help="Align points with 3D", default="False")
    parser.add_argument("--noise_std", type=float,
                        help="Noise level of inliers", default=0.0)

    parser.add_argument("--init_image1", type=int,
                        help="Image to leave unchanged", default=-1)
    parser.add_argument("--init_image2", type=int,
                        help="Image to leave unchanged", default=-1)

    parser.add_argument("--outlier_ratio", type=float,
                        help="Ratio of outliers in the total number of points.", default=0.0)
    parser.add_argument("--max_try_outlier", type=int,
                        help="Number of iterations to generate outliers.", default=100)

    args = parser.parse_args()

    print("CREATING DATABASE FROM FILES")
    main(args)
    print("GOING TO VERIFY")
    verify(args)
    if args.delete == "True":
        print("Deleting")
        if os.path.exists(args.database_path):
            os.remove(args.database_path)
