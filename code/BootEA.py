import sys
import time
import numpy as np

from train_funcs import get_model, generate_related_mat, train_tris_k_epo, train_alignment_1epo
from train_bp import bootstrapping, likelihood
from model import P

import utils as ut


def write_alignment(folder, model, rank_map):
    outpath = folder + '/rank_result'
    with open(outpath, 'w') as w:
        for kth, ranking in sorted(rank_map.items(), key=lambda x:x[0]):
            rank_list = ranking[:100]
            w.write('%d\t%s\n' % (np.int(kth), \
                ','.join([str(i) for i in rank_list.tolist()])))
    print('write final alignment:', outpath)


def train(folder):
    ori_triples1, ori_triples2, triples1, triples2, model = get_model(folder)
    hits1 = None
    rank_map = dict()

    labeled_align = set()
    ents1, ents2 = None, None

    # related_mat = generate_related_mat(folder, triples1, triples2, model.ref_ent1, model.ref_ent2)
    related_mat = None

    if P.epsilon > 0:
        trunc_ent_num = int(len(ori_triples1.ent_list) * (1 - P.epsilon))
        assert trunc_ent_num > 0
        print("trunc ent num:", trunc_ent_num)
    else:
        trunc_ent_num = 0
        assert not trunc_ent_num > 0
    if "DBP15K" in folder:
        for t in range(1, 50 + 1):
            print("iteration ", t)
            train_tris_k_epo(model, triples1, triples2, 5, trunc_ent_num, None, None)
            train_alignment_1epo(model, triples1, triples2, ents1, ents2, 1)
            train_tris_k_epo(model, triples1, triples2, 5, trunc_ent_num, None, None)
            labeled_align, ents1, ents2 = bootstrapping(model, related_mat, labeled_align)
            train_alignment_1epo(model, triples1, triples2, ents1, ents2, 1)
            hits1 = model.test(selected_pairs=labeled_align)
            likelihood(model, labeled_align)
            model.test(selected_pairs=labeled_align)
            ut.pair2file(folder + "results_BootEA_trunc" + str(P.epsilon), hits1)
    else:
        for t in range(1, 50 + 1):
            print("iteration ", t)
            train_tris_k_epo(model, triples1, triples2, 5, trunc_ent_num, None, None, is_test=False)
            train_alignment_1epo(model, triples1, triples2, ents1, ents2, 1)
            train_tris_k_epo(model, triples1, triples2, 5, trunc_ent_num, None, None, is_test=False)
            labeled_align, ents1, ents2 = bootstrapping(model, related_mat, labeled_align)
            train_alignment_1epo(model, triples1, triples2, ents1, ents2, 1)
            if t % 5 == 0 or t == 49:
                hits1, rank_map = model.test(selected_pairs=labeled_align)
    ut.pair2file(folder + "results_BootEA_trunc" + str(P.epsilon), hits1)
    model.save(folder, "BootEA_trunc" + str(P.epsilon))
    write_alignment(folder, model, rank_map)


if __name__ == '__main__':
    t = time.time()
    if len(sys.argv) == 2:
        folder = sys.argv[1]
    else:
        # folder = '../dataset/DBP15K/zh_en/mtranse/0_3/'
        folder = '../dataset/DWY100K/dbp_wd/mapping/0_3/'
    train(folder)
    print("total time = {:.3f} s".format(time.time() - t))
