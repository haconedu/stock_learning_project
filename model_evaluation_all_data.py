from data.corp import Corp
from trains.learning_n_mock_investment import LearningNMockInvestment
from params.train_params import TrainParams


def get_corps():
    """학습할 회사를 조회한다."""
    corp = Corp()
    return corp.get_eval_corps()


def train(start_no=1, params=None):
    """하나의 세션으로 학습시키는 기본 모델 """
    corps = get_corps()

    if params is None:
        params = TrainParams()
        params.is_all_corps_model = True
        params.result_file_name = 'training_invest_all_result'
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)

def train_deep(start_no=1, params=None):
    """하나의 세션으로 학습시키는 기본 모델 """

    corps = get_corps()

    if params is None:
        params = TrainParams()
        params.hidden_dims = [256, 128, 64, 32, 16]
        params.is_all_corps_model = True
        params.result_file_name = 'training_invest_deep_all_result'
        params.session_file_name = 'ALL_CORPS_DEEP'
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


def train_all_corps(start_no=1):
    """하나의 세션으로 모든 회사를 학습시킨다.  """
    corp = Corp()
    corps = corp.get_corps()

    params = TrainParams()
    params.is_all_corps_model = True
    params.result_file_name = 'training_all_corps'
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


def train_deep_all_corps(start_no=1, params=None):
    """조금 깊은 모델"""
    corp = Corp()
    corps = corp.get_corps()

    if params is None:
        params = TrainParams()
        params.hidden_dims = [256, 128, 64, 32, 16]
        params.is_all_corps_model = True
        params.result_file_name = 'training_deep_all_corps'
        params.session_file_name = 'ALL_CORPS_DEEP'
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


def train_up_down_to_one_session(start_no=1, params=None):
    """결과를 1,0으로 학습"""
    corps = get_corps()

    if params is None:
        params = TrainParams()
        params.is_all_corps_model = True
        params.result_file_name = 'training_invest_all_updown_result'
        params.session_file_name = 'ALL_CORPS_UPDOWN'
        params.y_is_up_down = True
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


if __name__ == '__main__':
    train_deep_all_corps(1424)
