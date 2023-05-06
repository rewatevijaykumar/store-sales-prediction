import sys

def error_message_detail(error,error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename

    error_message = f'Error occured in python scrit name: [{filename}] line number: [{exc_tb.tb_lineno}] error message: [{str(error)}]'

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        '''
        param error_message: error message in string format
        '''
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message,error_detail)
    
    def __str__(self):
        return self.error_message