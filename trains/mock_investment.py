import tensorflow as tf
import math
import numpy as np
from trains.learning import Learning
from models.stacked_rnn import StackedRnn


class MockInvestment:
    """모의투자"""

    def __init__(self, params):
        self.params = params

    def let_invest_money(self, invest_predict, now_scaled_close, now_close, now_money, now_stock_cnt):
        """예측 값에 따라 매수 매도를 실행한다."""
        if not self.params.y_is_up_down:
            return self.let_invest_money_value(invest_predict, now_scaled_close, now_close, now_money, now_stock_cnt)
        else:
            return self.let_invest_money_up_down(invest_predict, now_close, now_money, now_stock_cnt)

    def let_invest_money_value(self, invest_predict, now_scaled_close, now_close, now_money, now_stock_cnt):
        """예측 값에 따라 매수 매도를 실행한다."""
        invest_min_percent = self.params.invest_min_percent

        if now_scaled_close == 0:
            now_scaled_close = 0.000000001
        ratio = (invest_predict - now_scaled_close) / now_scaled_close * 100

        if ratio > invest_min_percent:
            now_money, now_stock_cnt = self.buy_stock(now_money, now_close, now_stock_cnt)
        elif ratio < -invest_min_percent:
            now_money, now_stock_cnt = self.sell_stock(now_money, now_close, now_stock_cnt)
        return now_money, now_stock_cnt

    def let_invest_money_up_down(self, invest_predict, now_close, now_money, now_stock_cnt):
        """예측 값에 따라 매수 매도를 실행한다."""
        if invest_predict > 0.51:
            now_money, now_stock_cnt = self.buy_stock(now_money, now_close, now_stock_cnt)
        elif invest_predict < 0.49 :
            now_money, now_stock_cnt = self.sell_stock(now_money, now_close, now_stock_cnt)
        return now_money, now_stock_cnt

    def buy_stock(self, now_money, now_close, now_stock_cnt):
        """주식을 산다."""
        fee_percent = self.params.fee_percent
        cnt = math.floor(now_money / now_close)
        if cnt > 0:
            fee = now_close * fee_percent / 100
            now_money -= (now_close + fee) * cnt
            now_stock_cnt += cnt
        return now_money, now_stock_cnt
    
    def sell_stock(self, now_money, now_close, now_stock_cnt):
        """주식을 판다."""
        if now_stock_cnt > 0:
            now_money += self.to_money(now_close, now_stock_cnt)
            now_stock_cnt = 0
        return now_money, now_stock_cnt

    def to_money(self, now_stock_cnt, now_close):
        """주식매도를 해서 돈으로 바꾼다."""
        money = 0
        if now_stock_cnt > 0:
            fee_percent = self.params.fee_percent
            tax_percent = self.params.tax_percent

            fee = now_close * fee_percent / 100
            tax = now_close * tax_percent / 100
            money = (now_close - (fee + tax)) * now_stock_cnt
        return money

    def get_real_money(self, data_params, scaler_close, last_predict):
        """실제 가격을 가져온다."""
        investRealCloses = data_params['investRealCloses']
        predict_money = scaler_close.inverse_transform(last_predict)
        last_close_money = investRealCloses[len(investRealCloses) - 1]
        last_pred_money = predict_money[0][0]
        return last_close_money, last_pred_money

    def let_invest(self, comp_code, dataX_last, data_params):
        """학습 후 모의 주식 거래를 한다."""
        stacked_rnn = StackedRnn(self.params)
        learning = Learning(self.params)
        invest_count = self.params.invest_count
        invest_money = self.params.invest_money
        investCloses = data_params['investCloses']
        investRealCloses = data_params['investRealCloses']
        investX = data_params['investX']

        graph_params = stacked_rnn.get_stacted_rnn_model()
        X = graph_params['X']
        Y_pred = graph_params['Y_pred']
        output_keep_prob = graph_params['output_keep_prob']
        session_file_path = learning.get_session_path(comp_code)
        now_stock_cnt = 0
        all_invest_money = invest_money
        all_stock_count = now_stock_cnt
        predicts = []
        now_close = 0
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, session_file_path)

            for i in range(invest_count):
                invest_predicts = sess.run(Y_pred, feed_dict={X: investX[i:i + 1], output_keep_prob: 1.0})
                predicts.append(invest_predicts[0])

                invest_predict = invest_predicts[0][0]
                now_scaled_close = investCloses[i][0]
                now_close = investRealCloses[i]
                #print(invest_predict, now_scaled_close, now_close)
                invest_money, now_stock_cnt = self.let_invest_money(invest_predict, now_scaled_close, now_close,
                                                                    invest_money, now_stock_cnt)
                all_invest_money, all_stock_count = self.let_invest_money(10.0, now_scaled_close, now_close,
                                                                          all_invest_money, all_stock_count)

            invest_money += self.to_money(now_stock_cnt, now_close)
            all_invest_money += self.to_money(all_stock_count, now_close)

            last_predict = sess.run(Y_pred, feed_dict={X: dataX_last, output_keep_prob: 1.0})
        # print(now_money)
        return invest_money, last_predict, predicts, all_invest_money

    def let_invest_and_all(self, comp_code, dataX_last, data_params, params_all):
        """학습 후 모의 주식 거래를 한다."""

        stacked_rnn = StackedRnn(self.params)
        learning = Learning(self.params)
        learning_all = Learning(params_all)

        invest_count = self.params.invest_count
        invest_money = self.params.invest_money

        # investX = data_params['investX']
        investCloses = data_params['investCloses']
        investRealCloses = data_params['investRealCloses']
        investX = data_params['investX']

        graph_params = stacked_rnn.get_stacted_rnn_model()
        X = graph_params['X']
        Y_pred = graph_params['Y_pred']
        output_keep_prob = graph_params['output_keep_prob']

        now_stock_cnt = 0

        file_path = learning.get_session_path(comp_code)
        file_path_all = learning_all.get_session_path(comp_code)

        all_invest_money = invest_money
        all_stock_count = 0
        now_close = 0
        predicts = []
        for i in range(invest_count):

            saver = tf.train.Saver()
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                saver.restore(sess, file_path)
                invest_predicts = sess.run(Y_pred, feed_dict={X: investX[i:i + 1], output_keep_prob: 1.0})
                invest_predict = invest_predicts[0][0]

            saver_all = tf.train.Saver()
            with tf.Session() as sess_all:
                init_all = tf.global_variables_initializer()
                sess_all.run(init_all)
                saver_all.restore(sess_all, file_path_all)
                invest_predicts_all = sess_all.run(Y_pred, feed_dict={X: investX[i:i + 1], output_keep_prob: 1.0})
                invest_predict_all = invest_predicts_all[0][0]

            invest_predict = np.mean([invest_predict, invest_predict_all])
            predicts.append([invest_predict])
            now_scaled_close = investCloses[i][0]
            now_close = investRealCloses[i]
            # print(invest_predict, now_scaled_close, now_close)
            invest_money, now_stock_cnt = self.let_invest_money(invest_predict, now_scaled_close, now_close,
                                                                invest_money, now_stock_cnt)
            if i == 0:
                all_invest_money, all_stock_count = self.let_invest_money(10.0, now_scaled_close, now_close,
                                                                          all_invest_money, all_stock_count)

        invest_money += self.to_money(now_stock_cnt, now_close)
        all_invest_money += self.to_money(all_stock_count, now_close)

        # 마지막 예측 값을 구한다.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, file_path)
            last_predict = sess.run(Y_pred, feed_dict={X: dataX_last, output_keep_prob: 1.0})

        saver_all = tf.train.Saver()
        with tf.Session() as sess_all:
            init_all = tf.global_variables_initializer()
            sess_all.run(init_all)
            saver_all.restore(sess_all, file_path_all)
            last_predict_all = sess_all.run(Y_pred, feed_dict={X: dataX_last, output_keep_prob: 1.0})

        last_predict = np.mean([last_predict, last_predict_all])

        return invest_money, last_predict, predicts, all_invest_money
