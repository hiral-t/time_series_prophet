import traceback


def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            tb = traceback.extract_tb(ex.__traceback__)
            filename, lineno, function_name, line = tb[-1]

            print(f"Exception in {filename}, line {lineno} in {function_name}: {ex}")
            raise ex

    return wrapper
