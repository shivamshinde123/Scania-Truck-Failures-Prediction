import logging



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

params = ReadParams().read_params()

model_creation_log_file_path = params['Log_paths']['model_creation']

file_handler = logging.FileHandler(model_creation_log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class MakeDataset:


    def __init__(self) -> None:
        pass

    