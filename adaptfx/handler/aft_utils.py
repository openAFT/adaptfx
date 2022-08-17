import time

def timing_with_time():
    start = time.perf_counter()
    time.sleep(1)
    stop = time.perf_counter()
    return (stop - start)