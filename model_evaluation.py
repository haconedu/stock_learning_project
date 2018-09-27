from data.corp import Corp
from trains.learning_n_mock_investment import LearningNMockInvestment
from params.train_params import TrainParams
from trains.learning_n_mock_top10 import LearningNMockTop10


def get_corps():
    corp = Corp()
    return corp.get_eval_corps()


def train(type='ALL_CORPS', start_no=1):
    """하나의 세션으로 학습시키는 기본 모델 """
    corps = get_corps()

    params = TrainParams(type)
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


def train_all_corps(type='ALL_CORPS', start_no=1):
    """하나의 세션으로 모든 회사를 학습시킨다.  """
    corp = Corp()
    corps = corp.get_corps()

    params = TrainParams(type)
    params.result_file_name = "training_" + type.lower() + "_result"
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


def train_up_down(start_no=1):
    """결과를 1,0으로 학습"""
    corps = get_corps()

    params = TrainParams()
    params.is_all_corps_model = True
    params.result_file_name = 'training_invest_all_updown_result'
    params.session_file_name = 'ALL_CORPS_UPDOWN'
    params.y_is_up_down = True
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


def top10_model():
    corp = Corp()
    corps = corp.get_eval_corps()

    params = TrainParams()
    params.invest_type = 'top10'
    params.result_file_name = "training_invest_top10_result"
    invests = LearningNMockTop10(params)
    invests.let_train_invests_top10(corps)


def twins(start_no=1):
    corp = Corp()
    corps = corp.get_eval_corps()

    params = TrainParams("EACH")
    params.result_file_name = 'twins_result'

    invests = LearningNMockInvestment(params)
    invests.let_train_invests_twins(corps, start_no)
    

if __name__ == '__main__':
    train()
