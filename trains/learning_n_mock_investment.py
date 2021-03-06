import pandas as pd
import numpy as np
import copy

from data.stocks import Stocks
from data.trains_data import TrainsData
from params.train_params import TrainParams
from trains.learning import Learning
from trains.mock_investment import MockInvestment
from data.data_utils import DataUtils
from data.corp import Corp


class LearningNMockInvestment:
    """학습시키고 모의투자를 실행한다."""

    # 학습 결과의 컬럼명 정의
    RESULT_COLUMNS = ['no', 'code', 'name', 'rmse', 'invest_result', 'all_invest_result', 'train_cnt']

    # 예측 결과의 컬럼명 정의
    RESULT_COLUMNS_NEXT =['no', 'last_date', 'code', 'name', 'rmse', 'train_cnt', 'last_close_money',
                                'last_pred_money', 'last_pred_ratio']

    def __init__(self, params):
        self.params = params

        if params.result_type == 'forcast':  # 예측의 경우
            self.result_columns = self.RESULT_COLUMNS_NEXT
        else:
            self.result_columns = self.RESULT_COLUMNS


    def let_train_invest(self, corp_code, corp_name, no):
        """입력한 회사에 대해서 학습시키고 모의투자를 실행한다."""

        stocks = Stocks(self.params)
        stock_data = stocks.get_stock_data(corp_code)

        invest_count =  self.params.invest_count

        if invest_count == 0:
            rmse_val, train_cnt, data_params, dataX_last, scaler_close = self.let_train_only(corp_code, stock_data)
            last_money = self.params.invest_money
            all_invest_money = last_money
        else:
            if self.params.is_all_corps_model == False and self.params.remove_session_file == True:
                learning = Learning(self.params)
                learning.delete_learning_image(corp_code)

            invest = MockInvestment(self.params)
            rmse_val, train_cnt, data_params, dataX_last, scaler_close = self.let_train_only(corp_code, stock_data)
            last_money, last_predict, invest_predicts, all_invest_money = invest.let_invest(corp_code, dataX_last,
                                                                                        data_params)

        if self.params.result_type == 'forcast':
            invest = MockInvestment(self.params)
            last_money, last_predict, invest_predicts, all_invest_money = \
                invest.let_invest(corp_code, dataX_last, data_params)
            last_date = stock_data.tail(1)['date'].to_string(index=False)
            last_close_money, last_pred_money = invest.get_real_money(data_params, scaler_close, last_predict)
            last_pred_ratio = (last_pred_money - last_close_money) / last_close_money * 100
            last_pred_ratio = "{:.2f}".format(last_pred_ratio) + "%"
            print(no, last_date, corp_code, corp_name, rmse_val, train_cnt, last_close_money, last_pred_money, last_pred_ratio)
            return [no, last_date, corp_code, corp_name, rmse_val, train_cnt, last_close_money, last_pred_money, last_pred_ratio]
        else:
            print(no, corp_code, corp_name, rmse_val, last_money, all_invest_money, train_cnt)
            return [no, corp_code, corp_name, rmse_val, last_money, all_invest_money, train_cnt]


    def let_train_invest_twins(self, corp_code, corp_name, no):
        """겨별 세션과 통합세션에서 예측한 값의 평균을 예측값으로 한다."""
        stocks = Stocks(self.params)
        trains_data = TrainsData(self.params)
        learning = Learning(self.params)
        params_all = TrainParams('ALL_CORPS')
        learning_all = Learning(params_all)
        invest = MockInvestment(self.params)

        stock_data = stocks.get_stock_data(corp_code)
        data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data)
        rmse_val, train_cnt, rmse_vals, test_predict = learning.let_learning(corp_code, data_params)
        rmse_val_all, train_cnt_all, rmse_vals_all, test_predict_all = learning_all.let_learning(corp_code, data_params)
        last_money, last_predict, invest_predicts, all_invest_money = \
            invest.let_invest_and_all(corp_code, dataX_last, data_params, params_all)
        rmse_val = np.mean([rmse_val, rmse_val_all])
        train_cnt = train_cnt + train_cnt_all

        print(no, corp_code, corp_name, rmse_val, last_money, all_invest_money, train_cnt)
        return [no, corp_code, corp_name, rmse_val, last_money, all_invest_money, train_cnt]


    def let_train_only(self, corp_code, stock_data, invest_count=None):
        """입력한 회사에 대해서 학습시킨다"""

        trains_data = TrainsData(self.params)
        learning = Learning(self.params)

        data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data, invest_count)
        rmse_val, train_cnt, rmse_vals, test_predict = learning.let_learning(corp_code, data_params)
        return rmse_val, train_cnt, data_params, dataX_last, scaler_close


    def let_train_invest_one(self, corp_code, stock_data):
        """입력한 회사에 대해서 학습시키고 한번의 모의투자를 실행한다."""
        invest = MockInvestment(self.params)

        rmse_val, train_cnt, data_params, dataX_last, scaler_close = self.let_train_only(corp_code, stock_data, 1)
        last_money, last_predict, invest_predicts, all_invest_money = invest.let_invest(corp_code, dataX_last, data_params)
        return rmse_val, last_money, all_invest_money, train_cnt

    def let_train_invests(self, corps, start_no=1):
        """입력한 회사들에 대해서 학습시키고 모의투자를 실행한다."""

        if self.params.is_all_corps_model == True and self.params.remove_session_file == True:
            learning = Learning(self.params)
            learning.delete_learning_image()

        comp_rmses = []
        no = 1
        for index, corp_data in corps.iterrows():
            if no < start_no:
                no += 1
                continue
            corp_code = corp_data['종목코드']
            corp_name = corp_data['회사명']
            try :
                result = self.let_train_invest(corp_code, corp_name, no)
            except Exception as inst:
                print(inst)
                no += 1
                continue

            comp_rmses.append(result)
            if no % 10 == 0:
                df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
                DataUtils.save_excel(df_comp_rmses, self.get_result_file_path())
            no += 1


    def let_train_invests_for_name(self, corp_names):
        """회사이름으로 검색하여 학습시킴 """
        corp = Corp()
        comp_rmses = []
        no = 1
        for corp_name in corp_names:
            corp_code = corp.get_corp_code(corp_name)
            result = self.let_train_invest(corp_code, corp_name, no)
            comp_rmses.append(result)
            no += 1
        df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
        DataUtils.save_excel(df_comp_rmses, self.get_result_file_path())


    def let_train_invests_twins(self, corps, start_no=1):
        """겨별 세션과 통합세션에서 예측한 값의 평균을 예측값으로 한다."""
        comp_rmses = []
        no = 1
        for index, corp_data in corps.iterrows():
            if no < start_no:
                no += 1
                continue
            corp_code = corp_data['종목코드']
            corp_name = corp_data['회사명']
            try :
                result = self.let_train_invest_twins(corp_code, corp_name, no)
            except Exception as inst:
                print(inst)
                no += 1
                continue

            comp_rmses.append(result)
            if no % 10 == 0:
                df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
                DataUtils.save_excel(df_comp_rmses, self.get_result_file_path())
            no += 1

    def get_result_file_path(self):
        """결과를 저장할 경로"""
        return './result/' + self.params.result_file_name + '.xlsx'

