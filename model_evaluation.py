from data.corp import Corp
from trains.learning_n_mock_investment import LearningNMockInvestment
from params.train_params import TrainParams
from trains.learning_n_mock_top10 import LearningNMockTop10


def main(start_no=1):
    corp = Corp()
    corps = corp.get_eval_corps()

    params = TrainParams()
    params.invest_type = 'top10'
    invests = LearningNMockInvestment(params)
    invests.let_train_invests(corps, start_no)


def top10():
    corp = Corp()
    corps = corp.get_eval_corps()

    params = TrainParams()
    params.invest_type = 'top10'
    params.result_file_name = "training_invest_top10_result"
    invests = LearningNMockTop10(params)
    invests.let_train_invests_top10(corps)

if __name__ == '__main__':
    top10()
