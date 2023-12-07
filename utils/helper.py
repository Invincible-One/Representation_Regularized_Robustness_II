def progress_bar(iteration, total, bar_length=30):
    progress = float(iteration) / total
    filled_length = int(bar_length * progress)
    bar = '*' * filled_length + '-' * (bar_length - filled_length)
    return f"[{bar}] {progress:.1%}"


def arg_intNchar(v):
    if v == 'None':
        return None
    return int(v) if v.isdigit() else v

def arg_floatNchar(v):
    if v == 'None':
        return None
    try:
        return float(v)
    except ValueError:
        return v
