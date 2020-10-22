import numpy as np
import heapq
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from layers import Dense, CrossCompressUnit

import metrics

def test_one_user_v2(rating, pos_items, Ks):
    r, auc = ranklist_by_sorted_v2(pos_items, rating, Ks)
    return get_performance(pos_items, r, auc, Ks)

def ranklist_by_sorted_v2(user_pos_test, rating, Ks):
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, rating, key=rating.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(rating, user_pos_test)
    return r, auc

def test_one_user(x, train_items, test_items, item_num, Ks):
    rating, u = x[0], x[1]
    training_items = train_items[u] if u in train_items else []
    user_pos_test = test_items[u]
    all_items = set(range(item_num))
    test_items = list(all_items - set(training_items))
    r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    return get_performance(user_pos_test, r, auc, Ks)

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

class MKR(object):
    def __init__(self, args, n_users, n_items, n_entities, n_relations, dataset):
        self.dataset = dataset
        self._parse_args(n_users, n_items, n_entities, n_relations)
        self._build_inputs()
        self._build_model(args)
        self._build_loss(args)
        self._build_train(args)

    def _parse_args(self, n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

        # for computing l2 loss
        self.vars_rs = []
        self.vars_kge = []

    def _build_inputs(self):
        self.user_indices = tf.placeholder(tf.int32, [None], 'user_indices')
        self.item_indices = tf.placeholder(tf.int32, [None], 'item_indices')
        self.labels = tf.placeholder(tf.float32, [None], 'labels')
        self.head_indices = tf.placeholder(tf.int32, [None], 'head_indices')
        self.tail_indices = tf.placeholder(tf.int32, [None], 'tail_indices')
        self.relation_indices = tf.placeholder(tf.int32, [None], 'relation_indices')

    def _build_model(self, args):
        self._build_low_layers(args)
        self._build_high_layers(args)

    def _build_low_layers(self, args):
        if os.path.exists('../data/' + self.dataset + '/user_embs_pretrain.npy'):
            user_emb_initializer = tf.constant_initializer(np.load('../data/' + self.dataset + '/user_embs_pretrain.npy'))
            self.user_emb_matrix = tf.get_variable('user_emb_matrix', [self.n_user, args.dim], initializer=user_emb_initializer)
            item_emb_initializer = tf.constant_initializer(np.load('../data/' + self.dataset + '/item_embs_pretrain.npy'))
            self.item_emb_matrix = tf.get_variable('item_emb_matrix', [self.n_item, args.dim], initializer=item_emb_initializer)
            ent_emb_initializer = tf.constant_initializer(np.load('../data/' + self.dataset + '/ent_embs_pretrain.npy'))
            self.entity_emb_matrix = tf.get_variable('entity_emb_matrix', [self.n_entity, args.dim], initializer=ent_emb_initializer)
            rel_emb_initializer = tf.constant_initializer(np.load('../data/' + self.dataset + '/rel_embs_pretrain.npy'))
            self.relation_emb_matrix = tf.get_variable('relation_emb_matrix', [self.n_relation, args.dim], initializer=rel_emb_initializer)
        else:
            self.user_emb_matrix = tf.get_variable('user_emb_matrix', [self.n_user, args.dim])
            self.item_emb_matrix = tf.get_variable('item_emb_matrix', [self.n_item, args.dim])
            self.entity_emb_matrix = tf.get_variable('entity_emb_matrix', [self.n_entity, args.dim])
            self.relation_emb_matrix = tf.get_variable('relation_emb_matrix', [self.n_relation, args.dim])

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        self.item_embeddings = tf.nn.embedding_lookup(self.item_emb_matrix, self.item_indices)
        self.head_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.head_indices)
        self.relation_embeddings = tf.nn.embedding_lookup(self.relation_emb_matrix, self.relation_indices)
        self.tail_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.tail_indices)
        self.cg_unit = CrossCompressUnit(args.dim)
        self.vals = []
        self.vectors = []

        for _ in range(args.L):
            user_mlp = Dense(input_dim=args.dim, output_dim=args.dim)
            tail_mlp = Dense(input_dim=args.dim, output_dim=args.dim)
            self.user_embeddings = user_mlp(self.user_embeddings)
            inp_v, inp_e = self.item_embeddings, self.head_embeddings
            self.vectors += [inp_v, inp_e]
            self.item_embeddings, self.head_embeddings, fake_v, fake_e = self.cg_unit([inp_v, inp_e])
            self.vectors += [self.item_embeddings, self.head_embeddings, fake_v, fake_e]
            #res_v, res_e, rfake_v, rfake_e = self.cg_unit([self.item_embeddings, self.head_embeddings])
            res_v, res_e, rfake_v, rfake_e = self.cg_unit([fake_v, fake_e])
            self.vectors += [res_v, res_e, rfake_v, rfake_e]
            self.tail_embeddings = tail_mlp(self.tail_embeddings)
            self.vals.append([inp_v, inp_e, fake_v, fake_e])

            self.vars_rs.extend(user_mlp.vars)
            self.vars_kge.extend(tail_mlp.vars)
        self.vars_rs.extend(self.cg_unit.vars)
        self.vars_kge.extend(self.cg_unit.vars)

    def _build_high_layers(self, args):
        # RS
        use_inner_product = True
        if use_inner_product:
            # [batch_size]
            self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        else:
            # [batch_size, dim * 2]
            self.user_item_concat = tf.concat([self.user_embeddings, self.item_embeddings], axis=1)
            for _ in range(args.H - 1):
                rs_mlp = Dense(input_dim=args.dim * 2, output_dim=args.dim * 2)
                # [batch_size, dim * 2]
                self.user_item_concat = rs_mlp(self.user_item_concat)
                self.vars_rs.extend(rs_mlp.vars)

            rs_pred_mlp = Dense(input_dim=args.dim * 2, output_dim=1)
            # [batch_size]
            self.scores = tf.squeeze(rs_pred_mlp(self.user_item_concat))
            self.vars_rs.extend(rs_pred_mlp.vars)
        self.scores_normalized = tf.nn.sigmoid(self.scores)

        # KGE
        # [batch_size, dim * 2]
        self.head_relation_concat = tf.concat([self.head_embeddings, self.relation_embeddings], axis=1)
        for _ in range(args.H - 1):
            kge_mlp = Dense(input_dim=args.dim * 2, output_dim=args.dim * 2)
            # [batch_size, dim]
            self.head_relation_concat = kge_mlp(self.head_relation_concat)
            self.vars_kge.extend(kge_mlp.vars)

        kge_pred_mlp = Dense(input_dim=args.dim * 2, output_dim=args.dim)
        # [batch_size, 1]
        self.tail_pred = kge_pred_mlp(self.head_relation_concat)
        self.vars_kge.extend(kge_pred_mlp.vars)
        self.tail_pred = tf.nn.sigmoid(self.tail_pred)

        self.scores_kge = tf.nn.sigmoid(tf.reduce_sum(self.tail_embeddings * self.tail_pred, axis=1))
        self.rmse = tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(self.tail_embeddings - self.tail_pred), axis=1) / args.dim))

    def _build_loss(self, args):
        # RS
        self.base_loss_rs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
        self.l2_loss_rs = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
        for var in self.vars_rs:
            self.l2_loss_rs += tf.nn.l2_loss(var)
        # self.loss_rs = self.base_loss_rs + self.l2_loss_rs * args.l2_weight
        self.cg_loss_rs = self.cg_unit.get_rs_loss(self.vals[0])
        for vecs in self.vals[1:]:
            self.cg_loss_rs += self.cg_unit.get_rs_loss(vecs)
        self.loss_rs = self.base_loss_rs + self.cg_loss_rs * args.cg_weight + self.l2_loss_rs * args.l2_weight

        # KGE
        self.base_loss_kge = -self.scores_kge
        self.l2_loss_kge = tf.nn.l2_loss(self.head_embeddings) + tf.nn.l2_loss(self.tail_embeddings)
        for var in self.vars_kge:
            self.l2_loss_kge += tf.nn.l2_loss(var)
        # self.loss_kge = self.base_loss_kge + self.l2_loss_kge * args.l2_weight
        self.cg_loss_kge = self.cg_unit.get_kge_loss(self.vals[0])
        for vecs in self.vals[1:]:
            self.cg_loss_kge += self.cg_unit.get_kge_loss(vecs)
        self.loss_kge = self.base_loss_kge + self.cg_loss_kge * args.cg_weight + self.l2_loss_kge * args.l2_weight

    def _build_train(self, args):
        # TODO: better optimizer?
        self.optimizer_rs = tf.train.AdamOptimizer(args.lr_rs).minimize(self.loss_rs)
        self.optimizer_kge = tf.train.AdamOptimizer(args.lr_kge).minimize(self.loss_kge)
        # self.optimizer_rs = tf.train.AdadeltaOptimizer(0.05).minimize(self.loss_rs)
        # self.optimizer_kge = tf.train.AdadeltaOptimizer(0.05).minimize(self.loss_kge)

    def train_rs(self, sess, feed_dict):
        return sess.run([self.optimizer_rs, self.loss_rs], feed_dict)

    def train_kge(self, sess, feed_dict):
        return sess.run([self.optimizer_kge, self.rmse], feed_dict)

    def eval_kge(self, sess, feed_dict):
        return sess.run(self.rmse, feed_dict)

    def save_vectors(self, dataset, sess, feed_dict):
        vals = sess.run([self.labels] + self.vectors, feed_dict)
        for i, val in enumerate(vals):
            np.save('npys/' + dataset + '/org_' + str(i) + '.npy', val)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        '''true_positives = sum([1 if p == 1 and l == 1 else 0 for p, l in zip(predictions, labels)])
        precision = true_positives / sum(predictions)
        recall = true_positives / sum(labels)
        f1 = 2 * precision * recall / (precision + recall)'''
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc#, precision, recall, f1

    def calc_ndcg_v2(self, sess, test_data, feed_dict):
        Ks = [5, 10, 20, 40, 50, 60, 80, 100]
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        user_dict, pos_dict, score_dict = {}, {}, {}
        for data, label, score in zip(test_data, labels, scores):
            uid, iid = data[0], data[1]
            if not uid in score_dict:
                score_dict[uid] = {}
            score_dict[uid][iid] = score
            if not uid in user_dict:
                user_dict[uid] = []
            user_dict[uid].append(iid)
            if label == 1:
                if not uid in pos_dict:
                    pos_dict[uid] = []
                pos_dict[uid].append(iid)
        n_test_users = len(user_dict.keys())
        for uid in score_dict.keys():
            scores = score_dict[uid]
            if uid not in pos_dict:
                continue
            poses = pos_dict[uid]
            re = test_one_user_v2(scores, poses, Ks)
            print(re)
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users



    def calc_ndcg(self, sess, model, train_data, test_data, batch_size):
        Ks = [5, 10, 20, 40, 50, 60, 80, 100]
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

        item_num = max(np.max(train_data[:, 1]), np.max(test_data[:, 1]))
        test_users = []
        train_items, test_items = {}, {}
        for uid, iid, label in train_data:
            if label == 1:
                if uid not in train_items:
                    train_items[uid] = []
                train_items[uid].append(iid)
        for uid, iid, label in test_data:
            if label == 1:
                if uid not in test_items:
                    test_items[uid] = []
                test_items[uid].append(iid)
                if uid not in test_users:
                    test_users.append(uid)
        n_test_users = len(test_users)

        n_item_batchs = item_num // batch_size + 1
        for i, uid in enumerate(test_users):
            if (i + 1) % 500 == 0:
                print("user:::", i, '/', len(test_users))
            item_batch = range(item_num)
            feed_dict = {model.user_indices: [uid] * item_num,
                    model.item_indices: item_batch,
                    model.labels: [1] * item_num,
                    model.head_indices: item_batch}
            rate_batch = sess.run(self.scores_normalized, feed_dict)

            re = test_one_user([rate_batch, uid], train_items, test_items, item_num, Ks)
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
        return result


    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
