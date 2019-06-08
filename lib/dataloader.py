import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import KFold

vocab = 'ACGT'
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_oligos_dataset(**kwargs):
    zipmap_version = kwargs.get('zipmap_version', '2018-12-14')
    threshold = kwargs.get('enrichment_threshold', 1000)  # in FPKM units
    subcellualr_target = kwargs.get('subcellular_target', 'CNN')
    load_secondary_structure = kwargs.get('load_secondary_structure', False)
    use_cross_validation = kwargs.get('use_cross_validation', False)

    print('Loading dataset zipmap version %s, %s fraction(s), with enrichment threshold set to %d' % (
        zipmap_version, subcellualr_target, threshold))

    if kwargs.get('seed') is not None:
        print('Setting seed to %d' % (kwargs.get('seed')))
        np.random.seed(kwargs.get('seed'))

    all_oligos_seq_path = os.path.join(basedir, 'data', 'oligos', '2017-08-04', 'All_oligos_2017-08-04.fa')
    all_oligos_struct_path = os.path.join(basedir, 'data', 'oligos', '2017-08-04', 'secondary_struct_2017-08-04.fa')
    raw_zipmap_dir_path = os.path.join(basedir, 'data', 'zipmap', 'processed', zipmap_version)

    if load_secondary_structure:
        # all oligos seq
        oligos_struct = {}
        with open(all_oligos_struct_path, 'r') as file:
            for row in file:
                if row.startswith('>'):
                    seq_id = row[1:].rstrip()
                else:
                    oligos_struct[seq_id] = row.rstrip()

    # all oligos seq
    oligos_seq = {}
    with open(all_oligos_seq_path, 'r') as file:
        for row in file:
            if row.startswith('>'):
                seq_id = row[1:].rstrip()
            else:
                oligos_seq[seq_id] = row.rstrip()
    ids = []
    seqs = []
    targets = []

    if zipmap_version == '2018-12-14':
        if subcellualr_target == 'CNN':
            # CNN: Cyto vs NucIns vs NucSol
            CNN_targets_file = pd.read_excel(
                os.path.join(raw_zipmap_dir_path, '%s_P2_Fractions_Normalized_Counts.xlsx' % (zipmap_version)))
            for i, row in CNN_targets_file.iterrows():
                row = row.values
                if not (row[0].startswith('ENST') or row[0].startswith('FB')):
                    continue
                if row[1:-4].mean() < threshold:
                    continue
                if row[0] not in oligos_seq.keys():
                    print('%s sequence not found' % (row[0]))
                ids.append(row[0])
                seqs.append(oligos_seq[row[0]])
                targets.append((lambda y: y / y.sum())(np.array([row[1:5].mean(), row[5:9].mean(), row[9:13].mean()])))
        elif subcellualr_target == 'MITO':
            # log2 fold change
            MITO_targets_file = pd.read_excel(
                os.path.join(raw_zipmap_dir_path,
                             '%s_Mitochondria_Mito_vs_Total_Diff_Expression.xlsx' % (zipmap_version)))
            for i, row in MITO_targets_file.iterrows():
                row = row.values
                if not (row[0].startswith('ENST') or row[0].startswith('FB')):
                    continue
                if row[1] < threshold:
                    continue
                if row[0] not in oligos_seq.keys():
                    print('%s sequence not found' % (row[0]))
                ids.append(row[0])
                seqs.append(oligos_seq[row[0]])
                targets.append(row[2])
        else:
            raise RuntimeError('Valid subcellular targets are \'MITO\' and \'CNN\'')
    elif zipmap_version == '2019-02-15':
        if subcellualr_target == 'GFP':
            # log2 fold change
            GFP_targets_file = pd.read_excel(
                os.path.join(raw_zipmap_dir_path,
                             '%s_%s_%s_vs_Total_Diff_Expression.xlsx' % (
                                 zipmap_version, subcellualr_target, subcellualr_target)))
            for i, row in GFP_targets_file.iterrows():
                row = row.values
                if not (row[0].startswith('ENST') or row[0].startswith('FB')):
                    continue
                if row[1] < threshold:
                    continue
                if row[0] not in oligos_seq.keys():
                    print('%s sequence not found' % (row[0]))
                ids.append(row[0])
                seqs.append(oligos_seq[row[0]])
                targets.append(row[2])
        else:
            raise RuntimeError('%s not supported for %s' % (subcellualr_target, zipmap_version))
    else:
        raise RuntimeError('Version %s not supported yet' % (zipmap_version))

    if load_secondary_structure:
        data = np.array(
            [np.array([vocab.index(c_n) + 4 * '.()'.index(c_s) for c_n, c_s in zip(seq, oligos_struct[seq_id])])
             for seq_id, seq in zip(ids, seqs)])
    else:
        data = np.array([np.array([vocab.index(c) for c in seq]) for seq in seqs])
    targets = np.array(targets)
    ids = np.array(ids)

    if len(targets.shape) == 1:
        targets = targets[:, None]

    total_size = len(data)

    permute = np.random.permutation(np.arange(total_size, dtype=np.int32))
    ids = ids[permute]
    data = data[permute]
    targets = targets[permute]

    if not use_cross_validation:
        test_ids = ids[-int(total_size * 0.1):]
        test_data = data[-int(total_size * 0.1):]
        test_targets = targets[-int(total_size * 0.1):]

        ids = ids[:-int(total_size * 0.1)]
        data = data[:-int(total_size * 0.1)]
        targets = targets[:-int(total_size * 0.1)]

        print('dataset size %d\ntraining set %d\ntest set %d' % (
            total_size, len(data), len(test_data)))

        return {
            'train_ids': ids,
            'train_data': data,
            'train_targets': targets,
            'test_ids': test_ids,
            'test_data': test_data,
            'test_targets': test_targets
        }
    else:
        kf = KFold(n_splits=5)
        splits = kf.split(data)
        return {
            'ids': ids,
            'data': data,
            'targets': targets,
            'splits': splits
        }


def load_kmer_features(**kwargs):
    zipmap_version = kwargs.get('zipmap_version', '2018-12-14')
    threshold = kwargs.get('enrichment_threshold', 1000)  # in FPKM units
    subcellualr_target = kwargs.get('subcellular_target', 'CNN')
    use_cross_validation = kwargs.get('use_cross_validation', False)

    print('Loading dataset zipmap version %s, %s fraction(s), with enrichment threshold set to %d' % (
        zipmap_version, subcellualr_target, threshold))

    if kwargs.get('seed') is not None:
        print('Setting seed to %d' % (kwargs.get('seed')))
        np.random.seed(kwargs.get('seed'))

    all_oligos_mers_path = os.path.join(basedir, 'data', 'oligos', '2017-08-04', '5Mers.csv')
    raw_zipmap_dir_path = os.path.join(basedir, 'data', 'zipmap', 'processed', zipmap_version)

    kmers = pd.read_csv(all_oligos_mers_path)

    '''IDs are not used'''
    all_ids = kmers['id'].to_list()
    del kmers['id']
    del kmers['length']

    '''get kmer features'''
    X = kmers.values

    ids = []
    features = []
    targets = []

    if zipmap_version == '2018-12-14':
        if subcellualr_target == 'CNN':
            # CNN: Cyto vs NucIns vs NucSol
            CNN_targets_file = pd.read_excel(
                os.path.join(raw_zipmap_dir_path, '%s_P2_Fractions_Normalized_Counts.xlsx' % (zipmap_version)))
            for i, row in CNN_targets_file.iterrows():
                row = row.values
                if not (row[0].startswith('ENST') or row[0].startswith('FB')):
                    continue
                if row[1:-4].mean() < threshold:
                    continue
                if row[0] not in all_ids:
                    print('%s sequence not found' % (row[0]))
                ids.append(row[0])
                features.append(X[all_ids.index(row[0])])
                targets.append((lambda y: y / y.sum())(np.array([row[1:5].mean(), row[5:9].mean(), row[9:13].mean()])))
        elif subcellualr_target == 'MITO':
            # log2 fold change
            MITO_targets_file = pd.read_excel(
                os.path.join(raw_zipmap_dir_path,
                             '%s_Mitochondria_Mito_vs_Total_Diff_Expression.xlsx' % (zipmap_version)))
            for i, row in MITO_targets_file.iterrows():
                row = row.values
                if not (row[0].startswith('ENST') or row[0].startswith('FB')):
                    continue
                if row[1] < threshold:
                    continue
                if row[0] not in all_ids:
                    print('%s sequence not found' % (row[0]))
                ids.append(row[0])
                features.append(X[all_ids.index(row[0])])
                targets.append(row[2])
        else:
            raise RuntimeError('Valid subcellular targets are \'MITO\' and \'CNN\'')
    elif zipmap_version == '2019-02-15':
        if subcellualr_target == 'GFP':
            # log2 fold change
            GFP_targets_file = pd.read_excel(
                os.path.join(raw_zipmap_dir_path,
                             '%s_%s_%s_vs_Total_Diff_Expression.xlsx' % (
                                 zipmap_version, subcellualr_target, subcellualr_target)))
            for i, row in GFP_targets_file.iterrows():
                row = row.values
                if not (row[0].startswith('ENST') or row[0].startswith('FB')):
                    continue
                if row[1] < threshold:
                    continue
                if row[0] not in all_ids:
                    print('%s sequence not found' % (row[0]))
                ids.append(row[0])
                features.append(X[all_ids.index(row[0])])
                targets.append(row[2])
        else:
            raise RuntimeError('%s not supported for %s' % (subcellualr_target, zipmap_version))
    else:
        raise RuntimeError('Version %s not supported yet' % (zipmap_version))

    data = np.stack(features, axis=0)
    targets = np.array(targets)
    ids = np.array(ids)

    print(data.shape)

    if len(targets.shape) == 1:
        targets = targets[:, None]

    total_size = len(data)

    permute = np.random.permutation(np.arange(total_size, dtype=np.int32))
    ids = ids[permute]
    data = data[permute]
    targets = targets[permute]

    if not use_cross_validation:
        test_ids = ids[-int(total_size * 0.1):]
        test_data = data[-int(total_size * 0.1):]
        test_targets = targets[-int(total_size * 0.1):]

        dev_ids = ids[-int(total_size * 0.2):-int(total_size * 0.1)]
        dev_data = data[-int(total_size * 0.2):-int(total_size * 0.1)]
        dev_targets = targets[-int(total_size * 0.2):-int(total_size * 0.1)]

        ids = ids[:-int(total_size * 0.2)]
        data = data[:-int(total_size * 0.2)]
        targets = targets[:-int(total_size * 0.2)]

        print('dataset size %d\ntraining set %d\nvalidation set %d\ntest set %d' % (
            total_size, len(data), len(dev_data), len(test_data)))

        return {
            'train_ids': ids,
            'train_data': data,
            'train_targets': targets,
            'dev_ids': dev_ids,
            'dev_data': dev_data,
            'dev_targets': dev_targets,
            'test_ids': test_ids,
            'test_data': test_data,
            'test_targets': test_targets
        }
    else:
        kf = KFold(n_splits=5)
        splits = kf.split(data)
        return {
            'ids': ids,
            'data': data,
            'targets': targets,
            'splits': splits
        }


if __name__ == "__main__":
    res = load_oligos_dataset(zipmap_version='2018-12-14', subcellular_target='CNN',
                              load_secondary_structure=False, enrichment_threshold=1)
    print(np.max(res['train_targets'], axis=0))
