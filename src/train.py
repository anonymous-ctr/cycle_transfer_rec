import os
import time
import random
import tensorflow as tf
import numpy as np
from model import MKR
from metrics import ndcg_at_k


def train(args, data, show_loss, show_topk):
    tf.set_random_seed(5)
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    # tidxes = np.arange(len(train_data))
    # np.random.shuffle(tidxes)
    # train_data = train_data[tidxes[:int(0.2 * len(train_data))]]
    kg = data[7]

    model = MKR(args, n_user, n_item, n_entity, n_relation, args.dataset)

    # top-K evaluation settings
    # user_num = 1000
    k_list = [2, 5, 10, 15, 20, 30, 50]
    train_record = get_user_record(train_data, True)
    eval_record = get_user_record(eval_data, False)
    test_record = get_user_record(test_data, False)
    eval_list = list(set(train_record.keys()) & set(eval_record.keys()))
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    # if len(user_list) > user_num:
    #     user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    # init_dir = "../weights/" + args.dataset + '_' + str(time.time())
    # os.mkdir(init_dir)
    
    with tf.Session() as sess:
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
        print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
        
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.save(sess, init_dir + "/model.ckpt")
        # saver.restore(sess, init_dir + "/model.ckpt")
        prev_acc = 0
        for step in range(args.n_epochs):
            # RS training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train_rs(sess, get_feed_dict_for_rs(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(loss)
    
            # KGE training
            if step % args.kge_interval == 0:
                np.random.shuffle(kg)
                start = 0
                while start < kg.shape[0]:
                    _, rmse = model.train_kge(sess, get_feed_dict_for_kge(model, kg, start, start + args.batch_size))
                    start += args.batch_size
                    if show_loss:
                        print(rmse)
            # CTR evaluation
            train_auc, train_acc = model.eval(sess, get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            train_rmse = model.eval_kge(sess, get_feed_dict_for_kge(model, kg, 0, train_data.shape[0]))
            eval_auc, eval_acc = model.eval(sess, get_feed_dict_for_rs(model, eval_data, 0, eval_data.shape[0]))
            eval_rmse = model.eval_kge(sess, get_feed_dict_for_kge(model, kg, 0, eval_data.shape[0]))
            test_auc, test_acc = model.eval(sess, get_feed_dict_for_rs(model, test_data, 0, test_data.shape[0]))
            test_rmse = model.eval_kge(sess, get_feed_dict_for_kge(model, kg, 0, test_data.shape[0]))
    
            #print('epoch %d    train auc: %.4f  acc: %.4f  pre %.4f  rec  %.4f, f1: %.4f\n\teval auc: %.4f  acc: %.4f pre: %.4f, rec: %.4f, f1: %.4f\n\ttest auc: %.4f  acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (step, train_auc, train_acc, train_pr, train_re, train_f1, eval_auc, eval_acc, eval_pr, eval_re, eval_f1, test_auc, test_acc, test_pr, test_re, test_f1))
            #print(train_rmse, eval_rmse, test_rmse)
            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f' % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

            '''if eval_acc > prev_acc:
                ent_embs = model.entity_emb_matrix.eval()
                rel_embs = model.relation_emb_matrix.eval()
                np.save('../data/' + args.dataset + '/ent_embs.npy', ent_embs)
                np.save('../data/' + args.dataset + '/rel_embs.npy', rel_embs)
                user_embs = model.user_emb_matrix.eval()
                item_embs = model.item_emb_matrix.eval()
                np.save('../data/' + args.dataset + '/user_embs.npy', user_embs)
                np.save('../data/' + args.dataset + '/item_embs.npy', item_embs)
                ng = model.cg_unit.g.eval()
                nf = model.cg_unit.f.eval()
                ngb = model.cg_unit.gb.eval()
                nfb = model.cg_unit.fb.eval()
                np.save('../data/' + args.dataset + '/weight_g.npy', ng)
                np.save('../data/' + args.dataset + '/weight_f.npy', nf)
                np.save('../data/' + args.dataset + '/bias_g.npy', ngb)
                np.save('../data/' + args.dataset + '/bias_f.npy', nfb)'''

            '''if eval_acc > prev_acc:
                user_embs = model.user_emb_matrix.eval()
                item_embs = model.item_emb_matrix.eval()
                np.save('../data/' + args.dataset + '/user_embs_pretrain.npy', user_embs)
                np.save('../data/' + args.dataset + '/item_embs_pretrain.npy', item_embs)
                ent_embs = model.entity_emb_matrix.eval()
                rel_embs = model.relation_emb_matrix.eval()
                np.save('../data/' + args.dataset + '/ent_embs_pretrain.npy', ent_embs)
                np.save('../data/' + args.dataset + '/rel_embs_pretrain.npy', rel_embs)'''

            # top-K evaluation
                
            '''if eval_acc > prev_acc:
                ent_embs = model.entity_emb_matrix.eval()
                rel_embs = model.relation_emb_matrix.eval()
                np.save('npys/' + args.dataset + '/org_ent_embs.npy', ent_embs)
                np.save('npys/' + args.dataset + '/org_rel_embs.npy', rel_embs)
                user_embs = model.user_emb_matrix.eval()
                item_embs = model.item_emb_matrix.eval()
                np.save('npys/' + args.dataset + '/org_user_embs.npy', user_embs)
                np.save('npys/' + args.dataset + '/org_item_embs.npy', item_embs)
                model.save_vectors(args.dataset, sess, get_feed_dict_for_rs(model, test_data, 0, test_data.shape[0]))'''
            
            '''if step > 40:
                precision, recall, f1, ndcg = topk_eval(
                    sess, model, user_list, train_record, eval_record, item_set, k_list)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()
                print('f1: ', end='')
                for i in f1:
                    print('%.4f\t' % i, end='')
                print()
                print('ndcg: ', end='')
                for i in ndcg:
                    print('%.4f\t' % i, end='')
                print('------------------\n')'''
           
            if show_topk and eval_acc > prev_acc:
                prev_acc = eval_acc
                # print('---> Eval set')
                # precision, recall, f1, ndcg = topk_eval(
                #     sess, model, eval_list, train_record, eval_record, item_set, k_list)
                # print('precision: ', end='')
                # for i in precision:
                #     print('%.4f\t' % i, end='')
                # print()
                # print('recall: ', end='')
                # for i in recall:
                #     print('%.4f\t' % i, end='')
                # print()
                # print('f1: ', end='')
                # for i in f1:
                #     print('%.4f\t' % i, end='')
                # print()
                # print('ndcg: ', end='')
                # for i in ndcg:
                #     print('%.4f\t' % i, end='')
                # print('\n')
                #
                # print('---> Test set')
                #t0 = time.time()
                precision, recall, f1, ndcg = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list)
                #print('...', time.time() - t0)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()
                print('f1: ', end='')
                for i in f1:
                    print('%.4f\t' % i, end='')
                print()
                print('ndcg: ', end='')
                for i in ndcg:
                    print('%.4f\t' % i, end='')
                print('\n')


def get_feed_dict_for_rs(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.head_indices: data[start:end, 1]}
    return feed_dict


def get_feed_dict_for_kge(model, kg, start, end):
    feed_dict = {model.item_indices: kg[start:end, 0],
                 model.head_indices: kg[start:end, 0],
                 model.relation_indices: kg[start:end, 1],
                 model.tail_indices: kg[start:end, 2]}
    return feed_dict


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    '''user_nums = []
    user_indices, item_indices, head_indices = [], [], []
    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        user_num = len(test_item_list)
        user_nums.append(user_num)
        user_indices += [user] * user_num
        item_indices += test_item_list
        head_indices += test_item_list
    all_items, all_scores = model.get_scores(sess, {model.user_indices: user_indices,
                                                model.item_indices: item_indices,
                                                model.head_indices: head_indices})
    cursor = 0
    for i, user in enumerate(user_list):
        next_cursor = cursor + user_nums[i]
        items = [all_items[k] for k in range(cursor, next_cursor)]
        scores = [all_scores[k] for k in range(cursor, next_cursor)]
        cursor = next_cursor
        item_score_map = dict()'''

    for user in user_list:
        if user not in test_record:
            continue
        # if len(test_record[user]) < 2:
        #     continue
        t0 = time.time()
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        items, scores = model.get_scores(sess, {model.user_indices: [user] * len(test_item_list),
                                                model.item_indices: test_item_list,
                                                model.head_indices: test_item_list})
        t1 = time.time()
        for item, score in zip(items, scores):
            item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        t2 = time.time()
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

            r = []
            for iid in item_sorted[:k]:
                if iid in test_record[user]:
                    r.append(1)
                else:
                    r.append(0)
            ndcg_list[k].append(ndcg_at_k(r, k))
        t3 = time.time()
        #print('///', t3 - t0, t1 - t0, t2 - t1, t3 - t2)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]

    return precision, recall, f1, ndcg


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
