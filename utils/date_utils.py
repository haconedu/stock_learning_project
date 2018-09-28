import datetime


class DateUtils:

    @staticmethod
    def to_date(date_str):
        """문자열을 데이트 형대로 변환한다."""
        date_str = date_str.replace(" ", "")
        split = ""
        if date_str.find("-") > -1:
            split = "-"
        elif date_str.find(".") > -1:
            split = "."
        date_format = '%Y' + split + '%m' + split + '%d'
        try :
            result = datetime.datetime.strptime(date_str, date_format)
        except Exception as inst:
            print(date_str)
            print(inst)
            result = datetime.datetime.now()
        return result
