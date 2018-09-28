
class TrainParams:
    """학습을 위한 파라미터를 정의한다."""

    def __init__(self, model_type='ALL_CORPS'):
        self.seq_length = 15  # 시퀀스 갯수
        self.data_dim = 5  # 입력 데이터 갯수
        self.dropout_keep = 0.6  # dropout
        self.output_dim = 1  # 출력 데이터 갯수
        self.learning_rate = 0.0001
        self.rmse_max = 0.05
        self.hidden_dims = [128, 96, 64, 32, 16]
        self.iterations = [100, 200]  # 최소, 최대 훈련 반복횟수
        self.loss_up_count = 50  # early stopping
        self.train_percent = 80.0  # 훈련 데이터 퍼센트
        self.invest_count = 20  # 투자 횟수
        self.invest_money = 10000000  # 각 주식에 모의투자할 금액
        self.fee_percent = 0.015  # 투자시 발생하는 수수료
        self.tax_percent = 0.3  # 매도시 발생하는 세금
        self.invest_min_percent = 0.315  # 투자를 하는 최소 간격 퍼센트
        self.kor_font_path = 'C:\\WINDOWS\\Fonts\\H2GTRM.TTF'
        self.remove_session_file = False
        self.is_all_corps_model = True
        self.result_type = 'default'
        self.invest_type = 'default'
        self.session_file_name = model_type
        self.result_file_name = model_type.lower() + '_result'
        self.remove_stock_days = 1

        if model_type == 'EACH':
            self.is_all_corps_model = False
        
        elif model_type == 'FORCAST':
            self.is_all_corps_model = True
            #self.session_file_name = 'ALL_CORPS'
            self.result_type = 'forcast'
            self.invest_count = 0
            self.remove_stock_days = 0

