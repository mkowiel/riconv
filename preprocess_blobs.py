import argparse
import glob
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


SEED = 31773


def save_h5(h5_filename, data, label, data_dtype='float16', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data',
        data=data,
        compression='gzip',
        compression_opts=4,
        dtype=data_dtype,
    )
    h5_fout.create_dataset(
        'label',
        data=label,
        compression='gzip',
        compression_opts=1,
        dtype=label_dtype,
    )
    h5_fout.close()


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def prepare_data_labels(filename='cmb_blob_labels.csv'):
    all_data_labels = pd.read_csv(filename, index_col='blob_map_filename')
    print(all_data_labels.head())

    labels = all_data_labels.ligand.unique()
    labels = np.sort(labels)
    labels = {label: i_label for i_label, label in enumerate(labels)}

    all_data_labels = all_data_labels.ligand.to_dict()
    return labels, all_data_labels


def save_labels(out_data_dir, labels):
    with open(os.path.join(out_data_dir, 'shape_names.txt'), 'w') as shape_names:
        for key, val in sorted(labels.items(), key=lambda x: x[1]):
            print(val, key)
            shape_names.write("%s\n" % key)


def extract_points_kmeans(filename, n_clusters, grid_size=2.0/283.0, verbose=0):
    start = datetime.now()
    with np.load(filename) as data:
        mask = data['blob'] > 0

        x, y, z = np.indices(data['blob'].shape)
        x = x - 0.5 * np.max(x)
        y = y - 0.5 * np.max(y)
        z = z - 0.5 * np.max(z)
        x = x[mask].reshape(-1, 1)
        y = y[mask].reshape(-1, 1)
        z = z[mask].reshape(-1, 1)
        xyz = np.hstack([x, y, z]) * grid_size
        xyz = xyz.astype('float16')
        # print('blob shape %s, non zero grid points: %d' %(data['blob'].shape, xyz.shape[0]))

        if xyz.shape[0] < n_clusters:
            xyz = np.tile(xyz, (1 + n_clusters // xyz.shape[0], 1))
            points = np.array(xyz[:n_clusters, :], dtype='float16')
            if verbose > 0:
                print("extract_points_kmeans: %.2fs, kmeans.fit(): %.2fs" % ((datetime.now() - start).total_seconds(), 0))
            return points

        start_kmeans = datetime.now()
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=3, max_iter=150, n_jobs=1)
        kmeans.fit(xyz)
        end_kmeans = datetime.now()
        points = np.array(kmeans.cluster_centers_, dtype='float16')
        np.random.shuffle(points)
        if verbose > 0:
            print("extract_points_kmeans: %.2fs, kmeans.fit(): %.2fs" % (
                (datetime.now() - start).total_seconds(), (end_kmeans - start_kmeans).total_seconds()
            ))
        return points


def run(in_data_dir, label_data_file_path, partial_data_dir, out_data_dir, file_name_prefix, n_points, n_bactch_size, verbose=0):
    np.random.seed(SEED)

    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    if not os.path.exists(partial_data_dir):
        os.makedirs(partial_data_dir)

    labels, all_data_labels = prepare_data_labels(label_data_file_path)
    save_labels(out_data_dir, labels)

    i_batch = 0
    point_data = []
    ligand_ids = []

    with open(os.path.join(out_data_dir, 'train_files.txt'), 'w') as train_files:
        npz_filenames = list(glob.glob(os.path.join(in_data_dir, "*.npz")))
        np.random.shuffle(npz_filenames)
        for filename in tqdm(npz_filenames):
            sample = os.path.basename(filename)
            if sample in all_data_labels:

                partial_full_path = os.path.join(partial_data_dir, sample.replace('.npz', '.h5'))
                if os.path.exists(partial_full_path):
                    points, ligand_id = load_h5(partial_full_path)
                    points = np.array(points[0], dtype='float16')
                    ligand_id = int(ligand_id[0])
                    if verbose > 0:
                        print(filename, partial_full_path, ligand_id)
                else:
                    ligand = all_data_labels[sample]
                    ligand_id = labels[ligand]
                    if verbose > 0:
                        print(filename, ligand, ligand_id)
                    points = extract_points_kmeans(filename, n_points, verbose=verbose)
                    save_h5(partial_full_path, [points], [ligand_id])

                point_data.append(points)
                ligand_ids.append(ligand_id)

                if len(point_data) >= n_bactch_size:
                    h5_filename = '%s_%03d.h5' % (file_name_prefix, i_batch)
                    train_files.write(h5_filename+'\n')
                    h5_filename = os.path.join(out_data_dir, h5_filename)
                    print('saving', h5_filename)
                    save_h5(h5_filename, point_data, ligand_ids)
                    i_batch += 1
                    point_data = []
                    ligand_ids = []

    if len(point_data) > 0:
        with open(os.path.join(out_data_dir, 'test_files.txt'), 'w') as test_files:
            h5_filename = '%s_%03d.h5' % (file_name_prefix, i_batch)
            test_files.write(h5_filename + '\n')
            h5_filename = os.path.join(out_data_dir, h5_filename)
            print('saving', h5_filename)
            save_h5(h5_filename, point_data, ligand_ids)

    print('END')


def fill_cache(in_data_dir, label_data_file_path, partial_data_dir, n_points, verbose=0):
    print('FILL CACHE MODE')
    if not os.path.exists(partial_data_dir):
        os.makedirs(partial_data_dir)

    labels, all_data_labels = prepare_data_labels(label_data_file_path)

    npz_filenames = list(glob.glob(os.path.join(in_data_dir, "*.npz")))
    np.random.shuffle(npz_filenames)
    for filename in tqdm(npz_filenames):
        sample = os.path.basename(filename)
        if sample in all_data_labels:

            partial_full_path = os.path.join(partial_data_dir, sample.replace('.npz', '.h5'))
            if not os.path.exists(partial_full_path):
                ligand = all_data_labels[sample]
                ligand_id = labels[ligand]
                if verbose > 0:
                    print(filename, ligand, ligand_id)
                points = extract_points_kmeans(filename, n_points, verbose=verbose)
                save_h5(partial_full_path, [points], [ligand_id])

    print('END')


parser = argparse.ArgumentParser()
parser.add_argument('--in_data_dir', default='blobs_full', help='Data dir [default: blobs_sample]')
parser.add_argument('--label_data_file_path', default='cmb_blob_labels.csv', help='data labels [default: cmb_blob_labels.csv]')
parser.add_argument('--partial_data_dir', default='partial', help='Cache data dir [default: partial]')
parser.add_argument('--out_data_dir', default='blobs_full_1024', help='Data dir [default: blobs_full_1024]')
parser.add_argument('--file_name_prefix', default='blobs_full', help='Data dir [default: blobs_full]')
parser.add_argument('--n_points', type=int, default=1024, help='number of points per sample [default: 1024]')
parser.add_argument('--n_bactch_size', type=int, default=2048, help='Number of samples in h5 file [default: 2048]')
parser.add_argument('--cache_only', type=bool, default=False, help='Save files only to cache dir [default: False]')
FLAGS = parser.parse_args()

if __name__ == "__main__":
    if FLAGS.cache_only:
        fill_cache(FLAGS.in_data_dir, FLAGS.label_data_file_path, FLAGS.partial_data_dir, FLAGS.n_points)
    else:
        run(FLAGS.in_data_dir, FLAGS.label_data_file_path, FLAGS.partial_data_dir, FLAGS.out_data_dir, FLAGS.file_name_prefix, FLAGS.n_points, FLAGS.n_bactch_size)
