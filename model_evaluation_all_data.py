import pandas as pd
from data.corp import Corp
from data.stocks import Stocks
from data.trains_data import TrainsData
from trains.learning import Learning
from trains.mock_investment import MockInvestment
from data.data_utils import DataUtils


def let_train_invest(corp_code, corp_name, params, no, session_file_name = 'ALL_CORPS', y_is_up_down=False):
    stocks = Stocks()
    trains_data = TrainsData(params, y_is_up_down)
    learning = Learning(params, True, session_file_name, y_is_up_down)
    invest = MockInvestment(params, True, session_file_name, y_is_up_down)

    stock_data = stocks.get_stock_data(corp_code)
    data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data)
    rmse_val, train_cnt, rmse_vals, test_predict = learning.let_learning(corp_code, data_params)
    last_money, last_predict, invest_predicts, all_invest_money = invest.let_invest(corp_code, train_cnt, dataX_last,
                                                                                    data_params)
    print(no, corp_code, corp_name, rmse_val, last_money, all_invest_money, train_cnt)
    return [no, corp_code, corp_name, rmse_val, last_money, all_invest_money, train_cnt]


def let_train_invests(corps, params, start_no=1, session_file_name='ALL_CORPS',
                      result_file_name='training_invest_all_result', y_is_up_down=False):
    comp_rmses = []
    no = 1
    for index, corp_data in corps.iterrows():
        if no < start_no:
            no += 1
            continue
        corp_code = corp_data['종목코드']
        corp_name = corp_data['회사명']
        result = let_train_invest(corp_code, corp_name, params, no, session_file_name, y_is_up_down)
        comp_rmses.append(result)
        if no % 10 == 0:
            df_comp_rmses = pd.DataFrame(comp_rmses,
                                         columns=['no', 'code', 'name', 'rmse', 'invest_result', 'all_invest_result',
                                                  'train_cnt'])
            DataUtils.save_excel(df_comp_rmses, './result/' + result_file_name + '.xlsx')
        no += 1


def get_basic_params():
    return {
            'seq_length': 5,  # 시퀀스 갯수
            'data_dim': 5,  # 입력 데이터 갯수
            'hidden_dims': [128, 96, 64],  # 히든 레이어 갯수
            'dropout_keep': 0.8,  # dropout
            'output_dim': 1,  # 출력 데이터 갯수
            'learning_rate': 0.0001,
            'iterations': [24, 100],  # 최소, 최대 훈련 반복횟수
            'rmse_max': 0.05,
            'train_percent': 80.0,  # 훈련 데이터 퍼센트
            'loss_up_count': 12,  # early stopping
            'invest_count': 20,  # 투자 횟수
            'invest_money': 10000000,  # 각 주식에 모의투자할 금액
            'fee_percent': 0.015,  # 투자시 발생하는 수수료
            'tax_percent': 0.5,  # 매도시 발생하는 세금
            'invest_min_percent': 0.6,  # 투자를 하는 최소 간격 퍼센트
            'kor_font_path': 'C:\\WINDOWS\\Fonts\\H2GTRM.TTF'
        }


def get_corps():
    corp = Corp()
    return corp.get_corps('1976-05-20', ['회사명', '종목코드'])


def train_to_one_session_deep(start_no=1, params=None, session_file_name='ALL_CORPS_DEEP',
                              result_file_name='training_invest_deep_all_result'):
    corps = get_corps()

    if params is None:
        params = get_basic_params()
        params['hidden_dims'] = [128, 128, 128, 128, 96, 96, 96, 64, 64]
    let_train_invests(corps, params, start_no, session_file_name, result_file_name)


def train_to_one_session_deep2(start_no=1, params=None, session_file_name='ALL_CORPS_DEEP2',
                              result_file_name='training_invest_deep2_all_result'):
    corps = get_corps()

    if params is None:
        params = get_basic_params()
        params['hidden_dims'] = [256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32]
    let_train_invests(corps, params, start_no, session_file_name, result_file_name)


def train_to_one_session(start_no=1, params=None, session_file_name='ALL_CORPS',
                         result_file_name='training_invest_all_result'):
    corps = get_corps()

    if params is None:
        params = get_basic_params()
    let_train_invests(corps, params, start_no, session_file_name, result_file_name)


def train_up_down_to_one_session(start_no=1, params=None, session_file_name='ALL_CORPS_UPDOWN',
                         result_file_name='training_invest_all_updown_result'):
    corps = get_corps()

    if params is None:
        params = get_basic_params().copy()
        params['iterations'] = [100, 10000]
        params['loss_up_count'] = 100
        params['hidden_dims'] = [512, 256, 128, 64, 32]
    let_train_invests(corps, params, start_no, session_file_name, result_file_name, True)


if __name__ == '__main__':
    train_up_down_to_one_session()
