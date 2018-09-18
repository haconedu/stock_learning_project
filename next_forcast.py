from trains.learning_n_mock_investment import LearningNMockInvestment
from params.train_params import TrainParams


def main(corp_names=None):

    if corp_names is None:
        corp_names = ["삼성중공업", "기아자동차", "게임빌", "루트로닉", "영진약품", "대아티아이"]
        #corp_names = ["JYP Ent.", "KTH"]

    params = TrainParams()
    params.is_all_corps_model = True
    params.result_file_name = 'next_forcast_result'
    params.result_type = 'forcast'
    params.invest_count = 0

    invests = LearningNMockInvestment(params)
    invests.let_train_invests_for_name(corp_names)


if __name__ == '__main__':
    main()
