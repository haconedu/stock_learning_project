from data.corp import Corp
from params.train_params import TrainParams
from trains.learning_n_mock_investment import LearningNMockInvestment


def main(start_no=1, params=None):
    corp = Corp()
    corps = corp.get_eval_corps()

    if params is None:
        params = TrainParams()
        params.result_file_name = 'training_invest_twins_result'

    invests = LearningNMockInvestment(params)
    invests.let_train_invests_twins(corps, start_no)


if __name__ == '__main__':
    main()
