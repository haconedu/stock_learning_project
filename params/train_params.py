

class TrainParams:
    """학습을 위한 파라미터를 정의한다."""

    def __init__(self):
        self.seq_length = 5  # 시퀀스 갯수
        self.data_dim = 5  # 입력 데이터 갯수
        self.hidden_dims = [128, 96, 64]  # 히든 레이어 갯수
        self.dropout_keep = 0.8  # dropout
        self.output_dim = 1  # 출력 데이터 갯수
        self.learning_rate = 0.0001
        self.iterations = [24, 120]  # 최소, 최대 훈련 반복횟수
        self.rmse_max = 0.05
        self.train_percent = 80.0  # 훈련 데이터 퍼센트
        self.loss_up_count = 12  # early stopping
        self.invest_count = 20  # 투자 횟수
        self.invest_money = 10000000  # 각 주식에 모의투자할 금액
        self.fee_percent = 0.015  # 투자시 발생하는 수수료
        self.tax_percent = 0.3  # 매도시 발생하는 세금
        self.invest_min_percent = 0.315  # 투자를 하는 최소 간격 퍼센트
        self.kor_font_path = 'C:\\WINDOWS\\Fonts\\H2GTRM.TTF'
        self.remove_session_file = False
        self.is_all_corps_model = False
        self.result_file_name = 'training_invest_result'
        self.session_file_name = 'ALL_CORPS'
        self.y_is_up_down = False  # 결과 값을 오르는지 내리는 지로 수정함
        self.result_type = 'default'
        self.invest_type = 'default'