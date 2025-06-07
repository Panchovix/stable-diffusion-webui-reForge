# This file is the main thread that handles all gradio calls for major t2i or i2i processing.
# Other gradio calls (like those from extensions) are not influenced.
# By using one single thread to process all major calls, model moving is significantly faster.


from __future__ import annotations

import queue
import threading
import traceback


lock = threading.Lock()
last_id = 0
task_queue: queue.Queue[Task] = queue.Queue()
task_dict: dict[int, Task] = {}


class Task:
    def __init__(self, task_id, func, args, kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.event = threading.Event()

    def work(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            traceback.print_exc()
            print(e)
        finally:
            self.event.set()


def loop():
    while True:
        task = task_queue.get()
        task.work()
        task_queue.task_done()


def async_run(func, *args, **kwargs):
    global last_id
    with lock:
        last_id += 1
        new_task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        task_dict[new_task.task_id] = new_task
    task_queue.put(new_task)
    return new_task.task_id


def run_and_wait_result(func, *args, **kwargs):
    current_id = async_run(func, *args, **kwargs)
    task = task_dict[current_id]
    task.event.wait()
    with lock:
        task_dict.pop(current_id, None)
    return task.result

